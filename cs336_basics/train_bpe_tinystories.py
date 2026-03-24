import os
from pathlib import Path
import json, base64
from typing import Sequence, Tuple, Dict, List

from tests.train_bpe import train_bpe

def _ensure_list_specials(special_tokens) -> List[str]:
    if not special_tokens:
        return []
    if isinstance(special_tokens, str):
        return [special_tokens]
    return list(special_tokens)

def _b2s(b: bytes) -> str:
    # bytes -> Base64 ASCII 字符串（JSON 安全）
    return base64.b64encode(b).decode("ascii")

def _s2b(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def save_tokenizer(
    out_dir: str | Path,
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    special_tokens: Sequence[str],
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) 保存词表（id -> bytes(base64)）
    with (out / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(
            {str(i): _b2s(tok) for i, tok in vocab.items()},
            f, ensure_ascii=False, indent=2
        )

    # 2) 保存 merges（[[a_b64, b_b64], ...]）
    with (out / "merges.json").open("w", encoding="utf-8") as f:
        json.dump([[ _b2s(a), _b2s(b) ] for (a,b) in merges], f, indent=2)

    # 3) 保存配置（可选）
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": len(vocab),
                "num_merges": len(merges),
                "special_tokens": list(special_tokens),
                "format": "byte_level_bpe_base64_v1"
            },
            f, ensure_ascii=False, indent=2
        )

def load_tokenizer(out_dir: str | Path):
    out = Path(out_dir)
    with (out / "vocab.json").open("r", encoding="utf-8") as f:
        v = json.load(f)
        vocab = {int(k): _s2b(v_) for k, v_ in v.items()}
    with (out / "merges.json").open("r", encoding="utf-8") as f:
        merges = [( _s2b(a), _s2b(b) ) for a,b in json.load(f)]
    with (out / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return vocab, merges, cfg

# -------- GPT-2 的 bytes<->unicode 可打印映射 --------
def _bytes_to_unicode():
    # 来自 OpenAI GPT-2 的做法：为 0..255 分配稳定、可打印的字符
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

def tok_bytes_to_str(bs: bytes, mode: str = "gpt2") -> str:
    """bytes -> 可读文本"""
    if mode == "gpt2":
        return "".join(_BYTE2UNI[b] for b in bs)
    elif mode == "escape":
        out = []
        for b in bs:
            if b == 0x0a: out.append("\\n")
            elif b == 0x09: out.append("\\t")
            elif b == 0x0d: out.append("\\r")
            elif 32 <= b <= 126 and b not in (34, 92):  # 可打印且不是 " 或 \
                out.append(chr(b))
            else:
                out.append(f"\\x{b:02x}")
        return "".join(out)
    else:
        raise ValueError("mode must be 'gpt2' or 'escape'")

def tok_str_to_bytes(s: str, mode: str = "gpt2") -> bytes:
    """可读文本 -> bytes（配套加载用；如果你只保存可先不需要）"""
    if mode == "gpt2":
        return bytes(_UNI2BYTE[ch] for ch in s)
    elif mode == "escape":
        # 简单解析：\xNN、\n、\t、\r 以及普通字符
        out = bytearray()
        i = 0
        while i < len(s):
            if s[i] == "\\" and i + 1 < len(s):
                if s[i+1] in "ntr":
                    out.append({ "n": 0x0a, "t": 0x09, "r": 0x0d }[s[i+1]])
                    i += 2
                elif s[i+1] == "x" and i + 3 < len(s):
                    out.append(int(s[i+2:i+4], 16))
                    i += 4
                else:
                    out.append(ord(s[i+1])); i += 2
            else:
                out.append(ord(s[i])); i += 1
        return bytes(out)
    else:
        raise ValueError("mode must be 'gpt2' or 'escape'")

# -------- 人类可读的保存/加载 --------

import json
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

def save_tokenizer_human(
    out_dir: str | Path,
    vocab: Dict[int, bytes],                    # id -> bytes
    merges: List[Tuple[bytes, bytes]],          # [(a_bytes, b_bytes), ...]
    special_tokens: Sequence[str],
    mode: str = "gpt2",                         # "gpt2" 或 "escape"
) -> None:
    """
    写出：
      - merges.txt  （第一行 '#version: 0.2'，每行 'tokA tokB'）
      - vocab.json  （token_string -> id 的映射，便于直观看）
      - config.json （元信息）
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) merges.txt：经典 BPE 形式
    merges_path = out / "merges.txt"
    with merges_path.open("w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in merges:
            sa = tok_bytes_to_str(a, mode)
            sb = tok_bytes_to_str(b, mode)
            # gpt2 模式下空格会是 'Ġ'，安全用空格分隔；escape 模式建议改成制表符
            sep = " " if mode == "gpt2" else "\t"
            f.write(f"{sa}{sep}{sb}\n")

    # 2) vocab.json：把 token 字符串作为 key，更直观
    #    （与 HuggingFace 的 byte-level BPE 习惯一致）
    vocab_path = out / "vocab.json"
    tok2id = { tok_bytes_to_str(bts, mode): tid for tid, bts in vocab.items() }
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(tok2id, f, ensure_ascii=False, indent=2)

    # 3) config.json：记录模式/大小/特殊符号
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "vocab_size": len(vocab),
                "num_merges": len(merges),
                "special_tokens": list(special_tokens or []),
                "human_format": mode,  # "gpt2" or "escape"
            },
            f, ensure_ascii=False, indent=2
        )

def load_tokenizer_human(out_dir: str | Path) -> tuple[Dict[int, bytes], List[Tuple[bytes, bytes]], dict]:
    """
    读回上述格式（如需）。注意 vocab.json 里是 token_string->id，需要倒置。
    """
    out = Path(out_dir)
    with (out / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    mode = cfg.get("human_format", "gpt2")

    with (out / "vocab.json").open("r", encoding="utf-8") as f:
        tok2id = json.load(f)
    vocab = { int(tid): tok_str_to_bytes(tok, mode) for tok, tid in tok2id.items() }

    # merges.txt
    merges: List[Tuple[bytes, bytes]] = []
    sep = " " if mode == "gpt2" else "\t"
    with (out / "merges.txt").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            sa, sb = line.split(sep, 1)
            merges.append((tok_str_to_bytes(sa, mode), tok_str_to_bytes(sb, mode)))

    return vocab, merges, cfg

if __name__ == "__main__":
    # data_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    data_path = "./data/owt_valid.txt"
    special_tokens = _ensure_list_specials("<|endoftext|>")
    vocab_size = 32_000

    # 训练（注意：special_tokens 必须是 list[str]）
    vocab, merges = train_bpe(
        input_path=data_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    # 序列化到硬盘
    ### version 1
    # save_tokenizer(
    #     out_dir="tokenizer_out_tiny_valid",
    #     vocab=vocab,
    #     merges=merges,
    #     special_tokens=special_tokens,
    # )
    #
    # #（可选）立刻读回校验一下
    # _v, _m, _cfg = load_tokenizer("tokenizer_out_tiny_valid")
    # assert len(_v) == len(vocab) and len(_m) == len(merges)
    # print("Saved tokenizer to ./tokenizer_out_tiny_valid")
    ### version 2
    save_tokenizer_human("tokenizer_out_tiny_owt", vocab, merges, special_tokens=['<|endoftext|>'], mode="gpt2")
