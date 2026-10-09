#!/usr/bin/env python3
"""Build D_fin **v2** (complete-information corpus) for the Gemma-4-E4B LGDSRFPA run.

v2 vs v1 (2026-10-10; + safe normalization, see normalize(), user decision: keep only data with complete information —
incomplete data hurts reasoning):
  * Completeness filter: drop chunks where >30% of non-table lines are bare numbers/symbols
    or the mean line length < 8 chars (broken PDF tables: one cell split over many lines,
    amounts detached from their row labels). In v1, ~77% of annual-report chunks were such fragments.
  * Annual reports re-extracted with table structure (annual_relayout.jsonl, pdfplumber
    table mode, rows as "label | amount | ...") REPLACE the old text-layer version of the same file.
  * Chunks are cut at line boundaries (never mid-row / mid-sentence).
  * Control chars (\x0c form feed etc.) removed; runs of layout spaces collapsed.

--- v1 docstring ---
Build D_fin for the Gemma-4-E4B LGDSRFPA run (ED §4.4 corpus-dose observation).

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
import unicodedata
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


CTRL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\ufffd]")
NUMLINE = re.compile(r"[\d,\.\-\(\)%\s$＄()（）]{1,25}")


def complete(c: str) -> bool:
    """True if the chunk carries complete information (not a shredded table)."""
    L = [l.strip() for l in c.splitlines() if l.strip()]
    if not L:
        return False
    plain = [l for l in L if "|" not in l]                  # relayout table rows are complete rows
    if plain:
        if sum(1 for l in plain if NUMLINE.fullmatch(l)) / len(plain) > 0.30:
            return False
        if sum(len(l) for l in plain) / len(plain) < 8:
            return False
    return True


FW_ALNUM = {c: c - 0xFEE0 for c in list(range(0xFF10, 0xFF1A)) + list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B))}
SPECIAL_SPACE = re.compile(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
ZERO_WIDTH = re.compile(r"[\u200b-\u200d\u2060\ufeff]")


def normalize(t: str) -> str:
    """Safe normalization (user request 2026-10-10). NOT NFKC: NFKC would turn Chinese
    full-width punctuation (，：（）) into ASCII. Only: NFC, full-width letters/digits → half-width,
    special/ideographic spaces → ' ', zero-width chars removed, CRLF → LF."""
    t = unicodedata.normalize("NFC", t)
    t = t.translate(FW_ALNUM)
    t = SPECIAL_SPACE.sub(" ", t)
    t = ZERO_WIDTH.sub("", t)
    t = re.sub(r"(?<=\d)，(?=\d{3}(?!\d))", ",", t)            # 123，456 → 123,456 (numbers only)
    t = re.sub(r"(?<=\d)．(?=\d)", ".", t)                      # 1．5 → 1.5
    return t.replace("\r\n", "\n").replace("\r", "\n")


def clean(t: str) -> str:
    t = normalize(t)
    t = t.replace("\x0c", "\n")
    t = CTRL.sub("", t)
    t = CJK_SPACE.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def chunk(t: str, size: int = CHUNK) -> list[str]:
    """Cut at line boundaries; a single over-long line is hard-split."""
    out, cur, n = [], [], 0
    for line in t.split("\n"):
        while len(line) > size:
            if cur:
                out.append("\n".join(cur)); cur, n = [], 0
            out.append(line[:size]); line = line[size:]
        if n + len(line) + 1 > size and cur:
            out.append("\n".join(cur)); cur, n = [], 0
        cur.append(line); n += len(line) + 1
    if cur:
        out.append("\n".join(cur))
    return [c.strip() for c in out if c.strip()]


def jtexts(pattern: str, exclude=(), skip_files=frozenset()):
    for fp in sorted(glob.glob(pattern, recursive=True)):
        if any(e in fp for e in exclude):
            continue
        with open(fp, errors="ignore") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if skip_files and r.get("file") in skip_files:
                    continue
                yield r.get("text", "")


RELAYOUT = C2.parent / "gemma3_corpus_v2" / "annual_relayout.jsonl"
SPACES = re.compile(r"[ \t]{3,}")


def collect(min_zh_len: int = 200) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []

    dropped: Counter = Counter()

    def add(texts, cat, minlen=300, zh=True):
        n0 = len(docs)
        for t in texts:
            t = clean(SPACES.sub(" ", t))
            if zh and not is_zh(t):
                continue
            for c in chunk(t):
                if len(c) >= minlen and (not zh or is_zh(c)):
                    if complete(c):
                        docs.append((c, cat))
                    else:
                        dropped[cat] += 1
        print(f"  {cat:9s} +{len(docs) - n0:,}  (dropped incomplete {dropped[cat]:,})", flush=True)

    add(jtexts(str(H / "gemma3_pretrain_data" / "**" / "*.jsonl"), exclude=("test_50mb",)),
        "news", minlen=min_zh_len)
    add(jtexts(str(C2 / "esg" / "esg-twse-*.jsonl")), "esg")
    add(jtexts(str(C2 / "esg_ocr" / "esg-ocr-*.jsonl")), "esg_ocr")
    add(jtexts(str(C2 / "esg_ocr_full" / "esg-ocr-*.jsonl")), "esg_ocr")
    relayout_rows = [json.loads(l) for l in open(RELAYOUT)] if RELAYOUT.exists() else []
    relayout_files = frozenset(r["file"] for r in relayout_rows)
    add(jtexts(str(C2 / "annual" / "annual-*.jsonl"), skip_files=relayout_files), "annual")
    add((r["text"] for r in relayout_rows), "annual_tbl")
    print(f"  (annual: {len(relayout_files)} reports replaced by table-preserving relayout)", flush=True)

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
    en = [c for t in jtexts(str(C2 / "edgar" / "edgar-clean-*.jsonl"))
          if len(t) > 300 and not is_zh(t)
          for c in chunk(clean(t))[:1] if complete(c)]
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
    ap.add_argument("--out", default=str(H / "gemma4_pretrain/data/dfin_v2"))
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
        upsample=upsample, heldout_excluded=n_held, seed=42, version="v2-complete-norm", normalization="NFC + fullwidth alnum→halfwidth + special spaces→space + zero-width removed + fullwidth digit separators in numbers→ASCII (no NFKC: keeps CJK punctuation)",
        filter="drop chunk if >30% non-table lines are bare numbers or mean line len < 8; relayout tables replace text-layer annual",
    )
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
