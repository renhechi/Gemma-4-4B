#!/usr/bin/env python3
"""Build D_fin for the Gemma-4-E4B LGDSRFPA run (ED §4.4 corpus-dose observation).

Differences from curated_zh_v5 (gemma3 track):
  * Reads the raw sources directly (annual reports have grown ~20x since v5).
  * Strips OCR/segmenter spaces between CJK chars ("安 能 聚" → "安能聚").
  * Global sha1 dedup; upsampling only for law / ins_law (×4) and penalty (×2), same as v5.
  * Excludes anything in ppl_heldout (FinKD held-out) to avoid contamination.
  * EDGAR English capped at ~12% of docs (v5 ratio).
  * Tokenises with the Gemma-4 tokenizer → uint32 tokens.bin + docs.npy index.

Split (fixed once, seed 42):
  * val  : ~0.5% of docs (min 600) — fixed L_val set, never trained on.
  * train: the rest, in a fixed shuffled order → the trainer consumes it sequentially,
           so "tokens seen" == corpus dose (0.25/0.50/0.75/1.00 D*).

Output: ~/gemma4_pretrain/data/dfin/{tokens.bin, docs.npy, meta.json}
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

H = Path.home()
C2 = H / "gemma3_corpus_v2"
CJK = re.compile(r"[一-鿿]")
# whitespace (not newline) sandwiched between CJK / fullwidth punctuation
CJK_SPACE = re.compile(r"(?<=[一-鿿　-〿＀-￯]) +(?=[一-鿿　-〿＀-￯])")
CHUNK = 3000          # chars; longer than v5 (1500) — 4096-token packing handles the rest


def is_zh(t: str, thr: float = 0.10) -> bool:
    n = min(len(t), 2000)
    return n > 0 and len(CJK.findall(t[:2000])) / n > thr


def clean(t: str) -> str:
    t = CJK_SPACE.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def chunk(t: str, size: int = CHUNK) -> list[str]:
    return [t[i:i + size] for i in range(0, len(t), size)]


def jtexts(pattern: str, exclude=()):
    for fp in sorted(glob.glob(pattern, recursive=True)):
        if any(e in fp for e in exclude):
            continue
        with open(fp, errors="ignore") as fh:
            for line in fh:
                try:
                    yield json.loads(line).get("text", "")
                except Exception:
                    pass


def collect(min_zh_len: int = 200) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []

    def add(texts, cat, minlen=300, zh=True):
        n0 = len(docs)
        for t in texts:
            t = clean(t)
            if zh and not is_zh(t):
                continue
            for c in chunk(t):
                if len(c) >= minlen and (not zh or is_zh(c)):
                    docs.append((c, cat))
        print(f"  {cat:9s} +{len(docs) - n0:,}", flush=True)

    add(jtexts(str(H / "gemma3_pretrain_data" / "**" / "*.jsonl"), exclude=("test_50mb",)),
        "news", minlen=min_zh_len)
    add(jtexts(str(C2 / "esg" / "esg-twse-*.jsonl")), "esg")
    add(jtexts(str(C2 / "esg_ocr" / "esg-ocr-*.jsonl")), "esg_ocr")
    add(jtexts(str(C2 / "esg_ocr_full" / "esg-ocr-*.jsonl")), "esg_ocr")
    add(jtexts(str(C2 / "annual" / "annual-*.jsonl")), "annual")

    def acad():
        for fp in glob.glob(str(H / "下載" / "files" / "邵汶" / "*.txt")):
            yield open(fp, encoding="utf-8", errors="ignore").read()
    add(acad(), "acad")

    add(jtexts(str(C2 / "news" / "*.jsonl")), "fin_news", minlen=min_zh_len)
    add(jtexts(str(C2 / "penalty" / "*.jsonl")), "penalty", minlen=min_zh_len)
    for name, cat in (("legal-tw.jsonl", "law"), ("insurance-tw.jsonl", "ins_law")):
        p = C2 / "legal" / name
        if p.exists():
            docs.extend((clean(json.loads(l)["text"]), cat) for l in open(p))
            print(f"  {cat:9s} (articles, no chunking)", flush=True)

    # EDGAR English: cap at 12% of total docs (v5 ratio)
    n_zh = len(docs)
    en_keep = int(n_zh * 0.12 / 0.88)
    en = [t[:CHUNK] for t in jtexts(str(C2 / "edgar" / "edgar-clean-*.jsonl"))
          if len(t) > 300 and not is_zh(t)]
    random.Random(42).shuffle(en)
    docs.extend((t, "edgar") for t in en[:en_keep])
    print(f"  {'edgar':9s} +{min(en_keep, len(en)):,} (of {len(en):,} available)", flush=True)
    return docs


def heldout_hashes() -> set[str]:
    hs = set()
    for fp in glob.glob(str(C2 / "ppl_heldout" / "*.jsonl")):
        for t in jtexts(fp):
            hs.add(hashlib.sha1(clean(t)[:500].encode()).hexdigest())
    return hs


_TOK = None


def _init(tok_path):
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tok_path)


def _encode(batch):
    enc = _TOK(batch, add_special_tokens=False)["input_ids"]
    return [np.asarray(e + [_TOK.eos_token_id], dtype=np.uint32) for e in enc]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(H / "gemma4_pretrain/models/gemma-4-E4B-it"))
    ap.add_argument("--out", default=str(H / "gemma4_pretrain/data/dfin"))
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[collect]", flush=True)
    docs = collect()
    held = heldout_hashes()
    upsample = {"law": 4, "ins_law": 4, "penalty": 2}
    seen: Counter = Counter()
    kept: list[tuple[str, str]] = []
    n_held = 0
    for t, c in docs:
        h = hashlib.sha1(t.encode()).hexdigest()
        if hashlib.sha1(t[:500].encode()).hexdigest() in held:
            n_held += 1
            continue
        if seen[h] >= 1:
            continue
        seen[h] += 1
        kept.extend([(t, c)] * upsample.get(c, 1))
    del docs
    print(f"[dedup] kept {len(kept):,} docs (heldout-excluded {n_held})", flush=True)

    rng = random.Random(42)
    rng.shuffle(kept)
    # val docs: never upsampled categories, fixed
    n_val = max(600, int(len(kept) * args.val_frac))
    val_idx, train_idx = [], []
    for i, (_, c) in enumerate(kept):
        (val_idx if len(val_idx) < n_val and c not in upsample else train_idx).append(i)
    order = val_idx + train_idx          # val first in tokens.bin

    print(f"[tok] encoding with {args.workers} workers …", flush=True)
    texts = [kept[i][0] for i in order]
    cats = [kept[i][1] for i in order]
    B = 256
    batches = [texts[i:i + B] for i in range(0, len(texts), B)]
    cat_names = sorted(set(cats))
    cat_id = {c: j for j, c in enumerate(cat_names)}
    index = np.zeros((len(texts), 4), dtype=np.int64)   # offset, length, cat_id, is_val
    tok_by_cat: Counter = Counter()
    off = 0
    k = 0
    with open(out / "tokens.bin", "wb") as fb, \
            mp.Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
        for bi, arrs in enumerate(pool.imap(_encode, batches, chunksize=4)):
            for a in arrs:
                fb.write(a.tobytes())
                index[k] = (off, len(a), cat_id[cats[k]], int(k < n_val))
                tok_by_cat[cats[k]] += len(a)
                off += len(a)
                k += 1
            if bi % 200 == 0:
                print(f"  {k:,}/{len(texts):,} docs  {off / 1e6:.1f}M tok", flush=True)
    np.save(out / "docs.npy", index)
    val_tok = int(index[:n_val, 1].sum())
    meta = dict(
        tokenizer=args.tokenizer, n_docs=len(texts), n_val_docs=n_val,
        total_tokens=int(off), val_tokens=val_tok, train_tokens=int(off - val_tok),
        tokens_by_cat=dict(tok_by_cat), cats=cat_names, chunk_chars=CHUNK,
        upsample=upsample, heldout_excluded=n_held, seed=42,
    )
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
