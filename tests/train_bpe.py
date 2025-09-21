from __future__ import annotations

import os
import mmap
import math
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Sequence, Tuple

# 用 regex 支持 \p{L}\p{N}
import regex as re
from collections import defaultdict

# =========================
# 常量与预分词（GPT-2 风格）
# =========================

# GPT-2 的预分词模式：保留空白团，英文足够用
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    flags=re.IGNORECASE,
)


def split_by_special_tokens(text: str, special_tokens: Sequence[str]) -> List[str]:
    """
    把文本按 special token 切开，并保留 special token 本体为独立片段。
    例如：
        text = "Hello <|endoftext|> world"
        -> ["Hello ", "<|endoftext|>", " world"]
    """
    if not special_tokens:
        return [text]
    # 按长度降序，避免短的 token 抢先匹配到长 token 的前缀
    specials = sorted(special_tokens, key=lambda s: -len(s))
    pat = re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")
    return pat.split(text)


def pretokenize(text: str, special_tokens: Sequence[str], drop_special: bool = True) -> List[bytes]:
    """
    预分词：
    - special token 分离为独立片段，，还是多个单词的组合
    - 对普通片段用 GPT-2 模式分词，分出单个单词
    - 返回“片段 token”的列表（每个 token 是 UTF-8 编码后的 bytes），decode后是字符串
    - 若 drop_special=True，则 special token **不会**进入返回列表（仅作为分隔符）
    """
    parts = split_by_special_tokens(text, special_tokens)
    out: List[bytes] = []
    for part in parts:
        if part in special_tokens:
            if not drop_special:
                out.append(part.encode("utf-8"))
            # 默认丢弃，不参与训练
            continue
        if not part:
            continue
        for s in GPT2_PAT.findall(part):
            if s:  # s 是 str
                out.append(s.encode("utf-8"))
    return out


# =========================
# 文件分块：对齐 special token
# =========================

def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_token_bytes: bytes,
) -> List[int]:
    """
    将文件切成若干块，**每个中间边界“前移”到下一个 split_token 出现位置**。
    这样可以保证 special token 不会被切成两半，便于并行预分词又不污染统计。
    返回每个边界的字节偏移（包含 0 和 file_size），去重升序。
    """
    assert isinstance(split_token_bytes, (bytes, bytearray))
    # 文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    # 初始均匀边界（最后一个强制是 file_size）
    chunk_size = file_size // desired_num_chunks
    guesses = [i * chunk_size for i in range(desired_num_chunks + 1)]
    guesses[-1] = file_size

    # 用内存映射快速扫描
    with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
        # 从第二个边界开始调整（首尾不动）
        for i in range(1, len(guesses) - 1):
            start = guesses[i]
            pos = mm.find(split_token_bytes, start)
            # 没找到就保持原位置（或退化到文件末尾）
            guesses[i] = file_size if pos == -1 else pos

    # 去重升序
    boundaries = sorted(set(guesses))
    # 至少保证 [0, file_size]
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    if boundaries[-1] != file_size:
        boundaries.append(file_size)
    return boundaries


# =========================
# BPE 训练：核心数据结构
# =========================
def _build_initial_vocab(special_tokens: Sequence[str]) -> Dict[int, bytes]:
    """
    初始词表：
    - 0..255 是所有单字节 token（bytes 直映）
    - 接着放入 special token（按给定顺序），确保它们在 vocab 中，但不参与训练
    """
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, st in enumerate(special_tokens):
        vocab[256 + i] = st.encode("utf-8")
    return vocab


# =========================
# 并行预分词（进程池）
# =========================

def _worker_pretokenize(args: Tuple[str, Tuple[str, ...]]) -> List[bytes]:
    text, specials = args
    return pretokenize(text, specials, drop_special=True)


def _parallel_pretokenize(
        chunks: List[str],
        special_tokens: Sequence[str],
        num_workers: int | None = None,
) -> List[bytes]:
    """
    多进程并行预分词，返回“片段 token 列表”，每个元素是 bytes。
    """
    if not chunks:
        return []
    workers = num_workers or max(1, (os.cpu_count() or 2))
    # 对于很小的数据，单进程反而更快
    if workers == 1 or sum(len(c) for c in chunks) < 1_000_000:
        out: List[bytes] = []
        for c in chunks:
            out.extend(pretokenize(c, special_tokens, drop_special=True))
        return out

    with mp.Pool(processes=workers) as pool:
        results = pool.map(_worker_pretokenize, [(c, tuple(special_tokens)) for c in chunks])
    out: List[bytes] = []
    for r in results:
        out.extend(r)
    return out


# =========================
# 统计相邻字节对（片段内部）
# =========================

def _count_adjacent_pairs(tokens: List[bytes]):
    """
    在“片段 token 的内部”统计相邻字节对。
    - tokens: List[bytes]，每个元素是一段 UTF-8 bytes（一个预分词片段）
    返回：
      counts: Dict[(int,int) -> int]         # (b1, b2) 出现次数
      index_map: Dict[(int,int) -> set[int]] # 该 pair 出现在哪些片段下标
    说明：
      - 初始阶段对“单字节对”计数（0..255），合并产生的新 token 用新的整数 id 表示（>=256+len(specials)）
      - 我们后续在 merge 时会把受影响的 tokens[i] 从 bytes 替换为 “int 列表”以便增量更新
    例子：
        输入是[b"Hello",...]
        b"Hello"就是bytes
    """
    counts = defaultdict(int)
    index_map = defaultdict(set)
    for idx, tok_bytes in enumerate(tokens):
        # 在 bytes 上迭代：每个元素是 0..255 的 int
        b = tok_bytes
        if len(b) < 2:
            continue
        for x, y in zip(b, b[1:]):  # x,y 均为 int(0..255)
            pair = (x, y)
            counts[pair] += 1
            index_map[pair].add(idx)
    return counts, index_map


# =========================
# 合并一步 & 增量更新
# =========================

def _merge_once_on_token(
        token: bytes | List[int],
        target_pair: Tuple[int, int],
        new_id: int,
) -> Tuple[List[int], List[int]]:
    """
    在单个“片段 token”内部执行一次合并，返回：
    - merged: 合并后的“int 序列”（以后都用 int 序列表示）
    - positions: 本次合并发生的位置（合并后序列的下标位置列表），用于增量更新相邻对计数
    说明：
    - 初始 token 若是 bytes，则把每个字节视为一个 int；合并后变成 int 序列。
    """
    if isinstance(token, (bytes, bytearray)):
        seq = list(token)  # 转成 0..255 的 int 序列
    else:
        seq = token

    a, b = target_pair
    n = len(seq)
    if n < 2:
        return (seq if isinstance(seq, list) else list(seq)), []

    merged: List[int] = []
    positions: List[int] = []
    i = 0
    pos = 0
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            merged.append(new_id)
            positions.append(pos)
            i += 2
        else:
            merged.append(seq[i])
            i += 1
        pos += 1
    return merged, positions


def _apply_merge_and_update(
    counts: Dict[Tuple[int, int], int],
    index_map: Dict[Tuple[int, int], set[int]],
    tokens: List[bytes | List[int]],
    target_pair: Tuple[int, int],
    new_id: int,
):
    """
    对包含 target_pair 的片段执行合并，并**增量**更新 counts/index_map。
    额外注意“相邻合并”的交叉对 (b, a)：当 ...ab ab... -> ...NEW NEW... 时，
    除了减去两次 (a,b)，还需要把中间那一次 (b,a) 再减 1。
    """
    affected = index_map.get(target_pair)
    if not affected:
        return

    a, b = target_pair
    affected_list = list(affected)

    # 先整体减掉该 pair 的出现次数（后续不会再看到旧的 (a,b)）
    total_occ = 0

    for idx in affected_list:
        old_tok = tokens[idx]
        merged, positions = _merge_once_on_token(old_tok, target_pair, new_id)
        if not positions:
            continue

        tokens[idx] = merged
        occ = len(positions)
        total_occ += occ

        # === 更新相邻对 ===
        # 说明：
        #   左侧：原本边界是 (LEFT, a)，若 LEFT 是刚刚合并出来的 NEW，则旧边界应视为 (b, a)
        #   右侧：原本边界是 (b, RIGHT)，若 RIGHT 是 NEW，则旧边界应视为 (b, a)
        #   新边界：变成 (LEFT, NEW) 与 (NEW, RIGHT)
        for p in positions:
            # 左邻
            if p - 1 >= 0:
                left = merged[p - 1]
                # 旧对减少
                if left == new_id:
                    old_left_pair = (b, a)         # 相邻合并交叉对
                else:
                    old_left_pair = (left, a)
                if old_left_pair in counts:
                    counts[old_left_pair] -= 1
                    if counts[old_left_pair] <= 0:
                        counts.pop(old_left_pair, None)
                        index_map.pop(old_left_pair, None)
                # 新对增加
                new_left_pair = (left, new_id)
                counts[new_left_pair] += 1
                index_map.setdefault(new_left_pair, set()).add(idx)

            # 右邻
            if p + 1 < len(merged):
                right = merged[p + 1]
                # 旧对减少
                if right == new_id:
                    old_right_pair = (b, a)        # 相邻合并交叉对
                else:
                    old_right_pair = (b, right)
                if old_right_pair in counts:
                    counts[old_right_pair] -= 1 ## 这里应该 处理index_map，但是好像也没啥问题，因为没用他来做判断。只用它来找affected index，如果已经删干净了，那肯定不会影响。
                    if counts[old_right_pair] <= 0:
                        counts.pop(old_right_pair, None)
                        index_map.pop(old_right_pair, None)
                # 新对增加
                new_right_pair = (new_id, right)
                counts[new_right_pair] += 1
                index_map.setdefault(new_right_pair, set()).add(idx)

    # 统一扣掉所有 (a,b)，统计了所有合并的ab，统一减去，如果小于等于0（应该最后发生等于0），就清理counts和index_map
    if total_occ:
        counts[target_pair] = counts.get(target_pair, 0) - total_occ
        if counts[target_pair] <= 0:
            counts.pop(target_pair, None)
            index_map.pop(target_pair, None)


# =========================
# 主训练流程
# =========================

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: Sequence[str] | None = None,
        num_workers: int | None = None,
        split_token_for_chunk: str = "<|endoftext|>",
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE：
    - vocab_size 为最终词表大小（含 256 基础字节 + special + 合并产生的 token）
    - special_tokens 仅加入词表；训练时**不统计也不参与合并**
    - 为了并行预分词，按 split_token_for_chunk 对齐切块（默认使用 endoftext 标记）
    """
    if vocab_size <= 256:
        raise ValueError("vocab_size must be > 256")
    specials = list(special_tokens or [])
    max_merges = max(0, vocab_size - 256 - len(specials))

    # 1) 初始 vocab
    vocab = _build_initial_vocab(specials)

    # 2) 切块（对齐 special），逐块读入并解码为 UTF-8 文本
    path = Path(input_path)
    chunks: List[str] = []
    with path.open("rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=(num_workers or max(1, (os.cpu_count() or 2))),
                                           split_token_bytes=split_token_for_chunk.encode("utf-8"))
        for s, e in zip(boundaries[:-1], boundaries[1:]):
            f.seek(s)
            chunks.append(f.read(e - s).decode("utf-8", errors="ignore"))

    # 3) 并行预分词，得到“片段 token”列表（bytes）
    tokens_bytes = _parallel_pretokenize(chunks, specials, num_workers=num_workers) ## 理论上这里每个单位都是一个词分成的bytes

    # 4) 初始统计相邻字节对（片段内部）
    counts, index_map = _count_adjacent_pairs(tokens_bytes)

    # 5) 合并循环（增量更新）
    merges: List[Tuple[bytes, bytes]] = []
    next_id = 256 + len(specials)

    # 为了在频次相同的情况下**确定性**，以 bytes 进行字典序比较（无需 decode）
    def _tie_key(item):
        (a, b), freq = item
        # 用“该 token 的实际字节串”来打破并列，稳定且不受 id 大小限制
        return (freq, vocab[a], vocab[b])

    # tokens 的容器：一开始每个元素是 bytes；一旦受影响，被替换为 List[int]
    tokens: List[bytes | List[int]] = list(tokens_bytes)

    for _ in range(max_merges):
        if not counts:
            break
        # 选出最高频 pair（频次，然后按 (a字节, b字节) 字典序稳定打破平手）
        target_pair, _ = max(counts.items(), key=_tie_key)

        a, b = target_pair
        # 记录 merges（以 bytes 形式）
        merges.append((vocab[a], vocab[b]))

        # 新 token id 与字节序列
        vocab[next_id] = vocab[a] + vocab[b]

        # 对受影响片段应用合并，并增量更新 counts/index_map
        _apply_merge_and_update(counts, index_map, tokens, target_pair, next_id)

        next_id += 1

    return vocab, merges
