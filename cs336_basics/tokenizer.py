from __future__ import annotations
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import json
from .train_bpe import pretokenize

# ===== GPT-2 bytes<->unicode 可打印映射（用于人类可读的 vocab/merges 文件） =====
def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_to_unicode = {b: chr(c) for b, c in zip(bs, cs)}
    unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}
    return byte_to_unicode, unicode_to_byte

_BYTE2UNI, _UNI2BYTE = _bytes_to_unicode()

def _tok_bytes_to_str(bs: bytes) -> str:
    """bytes -> GPT-2 可打印字符串（Ġ 表示空格等）。"""
    return "".join(_BYTE2UNI[b] for b in bs)

def _tok_str_to_bytes(s: str) -> bytes:
    """GPT-2 可打印字符串 -> 原始 bytes。"""
    return bytes(_UNI2BYTE[ch] for ch in s)

# ====== 预建查表 ======
def _build_fast_tables(
    merges: List[Tuple[bytes, bytes]],
    b2id: Dict[bytes, int],
) -> Tuple[List[int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    """
    返回：
      - byte2id_table: 256 长度 list，直接把单字节映射到初始 id
      - pair2new_id: (id_a, id_b) -> id_ab
      - pair2rank:   (id_a, id_b) -> 合并优先级 rank（越小越优先）
    """
    # 256 单字节表
    byte2id_table: List[int] = [0] * 256
    for i in range(256):
        bb = bytes([i])
        if bb not in b2id:
            raise KeyError(f"vocab 缺少单字节 {i} 的条目，无法进行 byte-level BPE")
        byte2id_table[i] = b2id[bb]

    pair2new_id: Dict[Tuple[int, int], int] = {}
    pair2rank: Dict[Tuple[int, int], int] = {}

    for rank, (a, b) in enumerate(merges):
        if a not in b2id or b not in b2id:
            raise KeyError(f"merges 中的 {a!r} 或 {b!r} 不在 vocab")
        ia = b2id[a]; ib = b2id[b]
        ab = a + b
        if ab not in b2id:
            raise KeyError(f"merge {a!r}+{b!r} 的结果 {ab!r} 不在 vocab")
        ic = b2id[ab]
        pair2new_id[(ia, ib)] = ic
        pair2rank[(ia, ib)] = rank

    return byte2id_table, pair2new_id, pair2rank

# ====== 参考语义版 BPE（与 tiktoken / OpenAI encoder 行为一致） ======
def _bpe_merge_reference_ints(
    ids: List[int],
    pair2new_id: Dict[Tuple[int, int], int],
    pair2rank: Dict[Tuple[int, int], int],
) -> List[int]:
    """
    反复：
      1) 找出当前序列中 rank 最小的相邻对；
      2) 从左到右把该对的所有**不重叠**出现一次性合并；
      3) 直到没有可合并对。
    与 tiktoken 的 bpe 语义一致（左到右、同 rank 不重叠）。
    """
    n = len(ids)
    if n <= 1:
        return ids

    # 为了少分配，循环里尽量重用局部变量
    while True:
        # 1) 找当前最小 rank 的相邻对
        best_pair: Optional[Tuple[int, int]] = None
        best_rank = 10**12  # 近似正无穷
        # 扫一遍相邻对
        for i in range(n - 1):
            p = (ids[i], ids[i + 1])
            r = pair2rank.get(p)
            if r is not None and r < best_rank:
                best_rank = r
                best_pair = p
        if best_pair is None:
            break  # 无可合并对

        new_id = pair2new_id[best_pair]

        # 2) 从左到右合并该对的所有不重叠出现
        out: List[int] = []
        i = 0
        last = n - 1
        a, b = best_pair
        while i < n:
            if i < last and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1

        # 更新 ids & n，继续下一轮
        ids = out
        n = len(ids)

    return ids

def _encode_pretoken_bytes_ref(
    pretok_bytes: bytes,
    byte2id_table: List[int],
    pair2new_id: Dict[Tuple[int, int], int],
    pair2rank: Dict[Tuple[int, int], int],
) -> List[int]:
    """把一个 pre-token（UTF-8 bytes）编码为 id 序列（参考语义 BPE）。"""
    ids = [byte2id_table[b] for b in pretok_bytes]
    if len(ids) <= 1:
        return ids
    return _bpe_merge_reference_ints(ids, pair2new_id, pair2rank)

# ====== Tokenizer ======
class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        vocab: id -> bytes
        merges: [(a_bytes, b_bytes), ...]   # 训练得到的 GPT-2 风格 merges
        special_tokens: 追加到 vocab（若不在的话），编码时把它们当单独 token，不参与合并
        """
        # --- 基础表 ---
        self.id2bytes: Dict[int, bytes] = dict(vocab)
        self.merges: List[Tuple[bytes, bytes]] = list(merges)
        self.special_tokens: List[str] = list(special_tokens or [])

        # 若用户提供的 special 不在 vocab 中，按顺序**追加**到 vocab
        existing = set(self.id2bytes.values())
        next_id = (max(self.id2bytes.keys()) + 1) if self.id2bytes else 0
        for st in self.special_tokens:
            sb = st.encode("utf-8")
            if sb not in existing:
                self.id2bytes[next_id] = sb
                existing.add(sb)
                next_id += 1

        # 反向映射
        self.bytes2id: Dict[bytes, int] = {b: i for i, b in self.id2bytes.items()}
        self._special_bytes: set[bytes] = {s.encode("utf-8") for s in self.special_tokens}

        # --- 高效表（一次性构建） ---
        self.byte2id_table, self.pair2new_id, self.pair2rank = _build_fast_tables(
            self.merges, self.bytes2id
        )

    # ----- 文件加载：人类可读格式（merges.txt + vocab.json） -----
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        """
        期望格式：
          - merges.txt：第一行可选 '#version: 0.2'，其后每行 'tokA tokB'（GPT-2 映射字符）
          - vocab.json：形如 {"Ġ":256, "t":5, ...} 的 token_string -> id 映射
        """
        # 1) 读取 vocab.json
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            tok2id = json.load(f)
        if not isinstance(tok2id, dict):
            raise ValueError("vocab.json must be a dict of token_string -> id")
        id2bytes: Dict[int, bytes] = {}
        for tok_str, tid in tok2id.items():
            if not isinstance(tid, int):
                raise ValueError("vocab.json values must be int ids")
            id2bytes[tid] = _tok_str_to_bytes(tok_str)

        # 2) 读取 merges.txt
        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid merges line: {line!r}")
                a, b = parts
                merges.append((_tok_str_to_bytes(a), _tok_str_to_bytes(b)))

        return cls(id2bytes, merges, special_tokens=special_tokens)

    # ----- 编码（内存内） -----
    def encode(self, text: str) -> List[int]:
        """
        预分词 -> 在每个 pre-token 内执行**参考语义 BPE 合并** -> 映射为 id。
        special token 原样输出其 id（不参与合并）。
        """
        # 注意：这里依赖你已有的 pretokenize(text, special_tokens, drop_special=False)
        pretoks: List[bytes] = pretokenize(text, self.special_tokens, drop_special=False)  # noqa: F821
        out: List[int] = []
        extend = out.extend
        b2id = self.bytes2id
        sp = self._special_bytes
        enc_one = _encode_pretoken_bytes_ref
        byte2id = self.byte2id_table
        p2n = self.pair2new_id
        p2r = self.pair2rank

        for tok in pretoks:
            if tok in sp:
                tid = b2id.get(tok)
                if tid is None:
                    raise KeyError(f"Special token not in vocab: {tok!r}")
                out.append(tid)
            else:
                extend(enc_one(tok, byte2id, p2n, p2r))
        return out

    # ----- 编码（流式，常量内存） -----
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        给出字符串可迭代（如文件句柄逐行），**懒惰地产生** token id。
        注意：不跨 pre-token 合并；若需跨块一致性，建议以文档分隔符对齐切块。
        """
        b2id = self.bytes2id
        sp = self._special_bytes
        enc_one = _encode_pretoken_bytes_ref
        byte2id = self.byte2id_table
        p2n = self.pair2new_id
        p2r = self.pair2rank

        for chunk in iterable:
            pretoks = pretokenize(chunk, self.special_tokens, drop_special=False)  # noqa: F821
            for tok in pretoks:
                if tok in sp:
                    tid = b2id.get(tok)
                    if tid is None:
                        raise KeyError(f"Special token not in vocab: {tok!r}")
                    yield tid
                else:
                    for tid in enc_one(tok, byte2id, p2n, p2r):
                        yield tid

    # ----- 解码 -----
    def decode(self, ids: List[int]) -> str:
        """
        id -> bytes 拼接 -> UTF-8 解码（errors='replace'，不合法序列替换为 U+FFFD）。
        """
        bb = bytearray()
        append = bb.extend
        id2b = self.id2bytes
        for tid in ids:
            b = id2b.get(tid)
            if b is None:
                append("\uFFFD".encode("utf-8"))
            else:
                append(b)
        return bytes(bb).decode("utf-8", errors="replace")