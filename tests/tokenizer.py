# =========================
# Tokenizer: encode / decode / streaming
# =========================
from typing import List, Tuple, Dict

from tests.train_bpe import pretokenize




import json
from typing import Iterator, Iterable, Optional

# --- GPT-2 bytes<->unicode 可打印映射（用于人类可读的 vocab/merges 文件） ---
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

# --- 合并应用在单个 pre-token 内（按 merges 顺序） ---
def _apply_merge_once_seq(seq: List[bytes], a: bytes, b: bytes) -> Tuple[List[bytes], bool]:
    """在 bytes 片段序列上应用一次 (a,b) 合并；返回(新序列, 是否有变化)。"""
    if not seq:
        return seq, False
    ab = a + b
    out: List[bytes] = []
    i, n = 0, len(seq)
    changed = False
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(ab)
            i += 2
            changed = True
        else:
            out.append(seq[i])
            i += 1
    return out, changed

def _encode_pretoken_bytes(
    pretok_bytes: bytes,
    merges: List[Tuple[bytes, bytes]],
    b2id: Dict[bytes, int],
) -> List[int]:
    """
    把一个 pre-token（UTF-8 bytes）编码为 id 序列：
    - 初始拆成单字节 [b'', ...]
    - 按 merges 的顺序在序列上进行合并
    - 把每个 bytes 片段映射为 id
    """
    seq: List[bytes] = [bytes([bt]) for bt in pretok_bytes]
    for a, b in merges:
        if len(seq) < 2:
            break
        seq, _ = _apply_merge_once_seq(seq, a, b)
    return [b2id[tok] for tok in seq]  # 若缺失会抛 KeyError，说明 vocab/merges 不一致

class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        vocab: id -> bytes
        merges: [(a_bytes, b_bytes), ...]   # 创建顺序即应用顺序
        special_tokens: 追加到 vocab（若不在的话），编码时把它们当单独 token，不参与合并
        """
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

    # ----- 文件加载：人类可读格式（merges.txt + vocab.json） -----
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        """
        期望格式（与我们建议的保存一致）：
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
                # 默认空格分隔（GPT-2 可打印映射中空格显示为 Ā/Ġ 等稀有字符，不会冲突）
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid merges line: {line!r}")
                a, b = parts
                merges.append((_tok_str_to_bytes(a), _tok_str_to_bytes(b)))

        return cls(id2bytes, merges, special_tokens=special_tokens)

    # ----- 编码（内存内） -----
    def encode(self, text: str) -> List[int]:
        """
        预分词 -> 在每个 pre-token 内按 merges 顺序合并 -> 映射为 id。
        special token 原样输出其 id（不参与合并）。
        """
        pretoks: List[bytes] = pretokenize(text, self.special_tokens, drop_special=False)
        out: List[int] = []
        for tok in pretoks:
            if tok in self._special_bytes:
                tok_id = self.bytes2id.get(tok)
                if tok_id is None:
                    raise KeyError(f"Special token not in vocab: {tok!r}")
                out.append(tok_id)
            else:
                out.extend(_encode_pretoken_bytes(tok, self.merges, self.bytes2id))
        return out

    # ----- 编码（流式，常量内存） -----
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        给出字符串可迭代（如文件句柄逐行），**懒惰地产生** token id。
        注意：不跨 pre-token 合并；若需跨块一致性，建议以文档分隔符对齐切块。
        """
        for chunk in iterable:
            pretoks = pretokenize(chunk, self.special_tokens, drop_special=False)
            for tok in pretoks:
                if tok in self._special_bytes:
                    tid = self.bytes2id.get(tok)
                    if tid is None:
                        raise KeyError(f"Special token not in vocab: {tok!r}")
                    yield tid
                else:
                    for tid in _encode_pretoken_bytes(tok, self.merges, self.bytes2id):
                        yield tid

    # ----- 解码 -----
    def decode(self, ids: List[int]) -> str:
        """
        id -> bytes 拼接 -> UTF-8 解码（errors='replace'，不合法序列替换为 U+FFFD）。
        """
        bb = bytearray()
        for tid in ids:
            b = self.id2bytes.get(tid)
            if b is None:
                bb.extend("\uFFFD".encode("utf-8"))
            else:
                bb.extend(b)
        return bytes(bb).decode("utf-8", errors="replace")
