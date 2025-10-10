# tokenizer_experiments.py
from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from array import array
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import regex as re

from .tokenizer import Tokenizer

# -------------------------
# GPT-2 style pretokenizer
# -------------------------
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    flags=re.IGNORECASE,
)


def split_by_special_tokens(text: str, special_tokens: Sequence[str]) -> List[str]:
    if not special_tokens:
        return [text]
    specials = sorted(special_tokens, key=lambda s: -len(s))
    pat = re.compile("(" + "|".join(re.escape(s) for s in specials) + ")")
    return pat.split(text)


def pretokenize(text: str, special_tokens: Sequence[str], drop_special: bool = True) -> List[bytes]:
    parts = split_by_special_tokens(text, special_tokens)
    out: List[bytes] = []
    for part in parts:
        if part in special_tokens:
            if not drop_special:
                out.append(part.encode("utf-8"))
            continue
        if not part:
            continue
        for s in GPT2_PAT.findall(part):
            if s:
                out.append(s.encode("utf-8"))
    return out


# --------------------------------------
# GPT-2 bytes<->unicode printable mapping
# --------------------------------------
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


def _tok_str_to_bytes(s: str) -> bytes:
    return bytes(_UNI2BYTE[ch] for ch in s)


# -------------------------
# Experiment helpers
# -------------------------
SPECIAL = "<|endoftext|>"


def load_docs_sample(path: str | Path, n_docs: int, seed: int = 0) -> List[str]:
    """Load file and sample n document strings; docs are split by SPECIAL if present else by blank lines."""
    p = Path(path)
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if SPECIAL in txt:
        parts = txt.split(SPECIAL)
    else:
        parts = [s for s in re.split(r"\n\s*\n", txt) if s.strip()]
    rng = random.Random(seed)
    if len(parts) <= n_docs:
        return [s for s in parts if s]
    idxs = rng.sample(range(len(parts)), n_docs)
    return [parts[i] for i in idxs if parts[i]]


def compression_ratio_bytes_per_token(texts: List[str], tok: Tokenizer) -> float:
    total_bytes = 0
    total_tokens = 0
    for s in texts:
        b = s.encode("utf-8")
        ids = tok.encode(s)
        total_bytes += len(b)
        total_tokens += max(1, len(ids))  # avoid div-by-zero on empty doc
    return total_bytes / total_tokens


def measure_throughput_bytes_per_sec(sample_path: str | Path, tok: Tokenizer) -> float:
    """Rough throughput by streaming the whole file."""
    start = time.time()
    total_bytes = 0
    with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            total_bytes += len(line.encode("utf-8"))
            # drain generator to simulate actual work
            for _ in tok.encode_iterable([line]):
                pass
    dt = max(1e-9, time.time() - start)
    return total_bytes / dt


def encode_corpus_to_npy(
        in_path: str | Path,
        out_npy: str | Path,
        tok: Tokenizer,
        dtype_if_possible: np.dtype = np.uint16,
) -> None:
    """Stream-encode a whole corpus to a single numpy array file (uint16 if vocab fits)."""
    print(f"start encoding {in_path}")
    vocab_size = max(tok.id2bytes.keys()) + 1 if tok.id2bytes else 0
    dtype = dtype_if_possible if vocab_size <= np.iinfo(dtype_if_possible).max else np.uint32
    ids: List[int] = []
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        for tid in tok.encode_iterable(f):
            ids.append(tid)
    arr = np.asarray(ids, dtype=dtype)
    np.save(out_npy, arr)


def encode_corpus_to_npy_fast(
        in_path: str | Path,
        out_npy: str | Path,
        tok,  # Tokenizer
        dtype_if_possible: np.dtype = np.uint16,
) -> None:
    """
    单进程、单遍编码：使用 array('H'/'I') 直接存原始数值，避免 Python list[int] 的对象开销。
    速度更快、内存更省（2/4 字节每 token）。
    """
    print(f"[fast] start encoding {in_path}")
    vocab_size = max(tok.id2bytes.keys()) + 1 if tok.id2bytes else 0
    dtype = dtype_if_possible if vocab_size <= np.iinfo(dtype_if_possible).max else np.uint32
    typecode = "H" if dtype == np.uint16 else "I"  # 'H'~uint16, 'I'~uint32（通常）

    buf = array(typecode)  # 紧凑二进制缓冲
    encode = tok.encode  # 局部绑定，少一次属性查找

    # i = 0
    start_time = time.perf_counter()
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.extend(encode(line))  # 一次性扩展一整行的 ids（比逐 token yield 更快）
            # i = i + 1
            # if i == 1000:
            #     print(f"[fast] early finished encoding {in_path}")
            #     break

    # array -> numpy（零拷贝视图）
    arr = np.frombuffer(buf, dtype=dtype)
    np.save(out_npy, arr)
    print(f"[fast] wrote {out_npy} with shape={arr.shape}, dtype={arr.dtype}, time cost:{time.perf_counter() - start_time}s")


# -------------------------
# CLI / Main
# uv run python -m tests.experiment_2_7.py --ts-tokenizer "./tokenizer_out_tiny_valid" --owt-tokenizer "./tokenizer_out_tiny_owt" --ts-train ./data/TinyStoriesV2-GPT4-train.txt --ts-dev ./data/TinyStoriesV2-GPT4-valid.txt --owt-train ./data/owt_train.txt --owt-dev ./data/owt_valid.txt
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Tokenizer experiments (a)-(d)")
    ap.add_argument("--ts-tokenizer", required=True, help="Dir with TinyStories vocab.json & merges.txt")
    ap.add_argument("--owt-tokenizer", required=True, help="Dir with OpenWebText vocab.json & merges.txt")
    ap.add_argument("--ts-train", required=True)
    ap.add_argument("--ts-dev", required=True)
    ap.add_argument("--owt-train", required=True)
    ap.add_argument("--owt-dev", required=True)
    ap.add_argument("--sample-docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ts_dir = Path(args.ts_tokenizer)
    owt_dir = Path(args.owt_tokenizer)

    # load tokenizers (assume human-readable files as discussed)
    ts_tok = Tokenizer.from_files(ts_dir / "vocab.json", ts_dir / "merges.txt", special_tokens=[SPECIAL])
    owt_tok = Tokenizer.from_files(owt_dir / "vocab.json", owt_dir / "merges.txt", special_tokens=[SPECIAL])

    # (a) sample & compression ratios
    # ts_docs = load_docs_sample(args.ts_train, args.sample_docs, seed=args.seed)
    # owt_docs = load_docs_sample(args.owt_train, args.sample_docs, seed=args.seed)
    #
    # cr_ts_on_ts = compression_ratio_bytes_per_token(ts_docs, ts_tok)
    # cr_owt_on_owt = compression_ratio_bytes_per_token(owt_docs, owt_tok)

    # print("(a) TinyStories tokenizer (10K) on TinyStories sample: "
    #       f"{cr_ts_on_ts:.3f} bytes/token. "
    #       "OpenWebText tokenizer (32K) on OWT sample: "
    #       f"{cr_owt_on_owt:.3f} bytes/token.")
    #
    # # (b) cross-tokenize OWT with TinyStories tokenizer
    # cr_ts_on_owt = compression_ratio_bytes_per_token(owt_docs, ts_tok)
    # print("(b) Tokenizing OWT with the TinyStories tokenizer yields "
    #       f"{cr_ts_on_owt:.3f} bytes/token vs {cr_owt_on_owt:.3f} with the OWT tokenizer; "
    #       "compression is worse because the TinyStories merges/vocab are mismatched to web text.")

    # (c) throughput & Pile time estimate (825 GB)
    # bps_ts = measure_throughput_bytes_per_sec(args.ts_dev, ts_tok)
    # seconds_for_pile = (825 * (1024**3)) / max(1.0, bps_ts)
    # hours = seconds_for_pile / 3600.0
    # print("(c) Estimated throughput:", f"{bps_ts:,.0f} bytes/s.",
    #       f"Tokenizing The Pile (825GB) would take ~{hours:.1f} hours on this machine with the current implementation.")

    # (d) encode all datasets to npy (uint16 when possible)
    out_ts_train = Path("ts_train_ids.npy")
    out_ts_dev = Path("ts_dev_ids.npy")
    out_owt_train = Path("owt_train_ids.npy")
    out_owt_dev = Path("owt_dev_ids.npy")

    encode_corpus_to_npy_fast(args.ts_dev, out_ts_dev, ts_tok) # lines: 157_832
    encode_corpus_to_npy_fast(args.ts_train, out_ts_train, ts_tok) # lines: 15_600_057
    encode_corpus_to_npy_fast(args.owt_dev,  out_owt_dev,  owt_tok) # lines: 2_301_019
    encode_corpus_to_npy_fast(args.owt_train,out_owt_train,owt_tok) # lines: 94_568_885

    # dtype explanation
    max_vocab_ts = max(ts_tok.id2bytes.keys()) + 1
    max_vocab_owt = max(owt_tok.id2bytes.keys()) + 1
    uses_uint16_ts = max_vocab_ts <= np.iinfo(np.uint16).max
    uses_uint16_owt = max_vocab_owt <= np.iinfo(np.uint16).max
    print("(d) We serialize IDs as NumPy uint16 because vocab sizes (e.g., "
          f"{max_vocab_ts} and {max_vocab_owt}) fit within 65,535; uint16 halves storage vs uint32 "
          "and speeds I/O/caching while preserving exact values.")


if __name__ == "__main__":
    main()
