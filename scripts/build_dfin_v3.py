#!/usr/bin/env python3
"""Build D_fin **v3**: prose kept, broken tables EXCISED, every remaining table complete.

v3 vs v2 (2026-10-13, user: 「有表格的部分一定要完整的表格,而不是全部的 data 僅是表格」):
  * v2 judged whole chunks (<30% shredded lines ⇒ kept), so shredded table fragments survived
    inside prose chunks (~35% of v2 tokens). v3 cuts out only the shredded-table segments
    (excise()) and keeps the surrounding prose — also recovering prose from chunks v2 dropped.
  * Table blocks (consecutive "a | b | c" rows from the relayout extraction) are never split;
    tables longer than CHUNK are split by rows with the header row repeated (each piece complete).
  * Final check: a chunk with any run of ≥3 fragment lines is dropped.

--- v2 docstring ---
Build D_fin **v2** (complete-information corpus) for the Gemma-4-E4B LGDSRFPA run.

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


PAGE = re.compile(r"[-–—]?\s*\d{1,4}\s*[-–—]?|第\s*\d+\s*頁|\d+\s*/\s*\d+")
SENT_END = re.compile(r"[。！？；!?;]")


def _kind(l: str) -> str:
    s = l.strip()
    if not s:
        return "B"                                            # blank
    if "|" in s:
        return "T"                                            # complete table row (relayout)
    if NUMLINE.fullmatch(s) or len(s) <= 3:
        return "F"                                            # shredded cell: bare number/symbol/1–3 chars
    if len(s) <= 12 and not SENT_END.search(s):
        return "S"                                            # short line: typical shredded header
    return "P"                                                # prose


def excise(t: str) -> str:
    """Remove shredded-table segments: maximal runs of F/S/B lines with ≥4 F lines. Page markers dropped."""
    lines = [l for l in t.split("\n") if not PAGE.fullmatch(l.strip())]
    kinds = [_kind(l) for l in lines]
    out, i, n = [], 0, len(lines)
    while i < n:
        if kinds[i] in "FS":
            j = i
            while j < n and kinds[j] in "FSB":
                j += 1
            if sum(k == "F" for k in kinds[i:j]) >= 4:
                out.append("")                                # drop the shredded block
            else:
                out.extend(lines[i:j])
            i = j
        else:
            out.append(lines[i]); i += 1
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def no_fragments(c: str) -> bool:
    run = 0
    for l in c.splitlines():
        k = _kind(l)
        if k == "B":
            continue
        run = run + 1 if k == "F" else 0
        if run >= 3:
            return False
    return True


def chunk_tables(t: str, size: int = CHUNK) -> list[str]:
    """Pack prose lines and whole table blocks into chunks ≤ size; never split a table
    unless it alone exceeds size, then split by rows repeating the header row."""
    blocks, cur, cur_tbl = [], [], None
    for l in t.split("\n"):
        is_t = "|" in l
        if cur and is_t != cur_tbl:
            blocks.append((cur_tbl, cur)); cur = []
        cur_tbl = is_t; cur.append(l)
    if cur:
        blocks.append((cur_tbl, cur))
    units = []                                                # (text, is_table) units that must stay whole
    for is_t, ls in blocks:
        if not is_t:
            units += [(c, False) for c in chunk("\n".join(ls), size)]
            continue
        txt = "\n".join(ls)
        if len(txt) <= size:
            units.append((txt, True)); continue
        head, piece = ls[0], [ls[0]]
        for row in ls[1:]:
            if len("\n".join(piece)) + len(row) + 1 > size and len(piece) > 1:
                units.append(("\n".join(piece), True)); piece = [head]
            piece.append(row)
        if len(piece) > 1:
            units.append(("\n".join(piece), True))
    out, buf, n = [], [], 0
    for txt, _ in units:
        if buf and n + len(txt) + 1 > size:
            out.append("\n".join(buf)); buf, n = [], 0
        buf.append(txt); n += len(txt) + 1
    if buf:
        out.append("\n".join(buf))
    return [c.strip() for c in out if c.strip()]


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
            t = excise(clean(SPACES.sub(" ", t)))
            if zh and not is_zh(t):
                continue
            for c in chunk_tables(t):
                if len(c) >= minlen and (not zh or is_zh(c)):
                    if no_fragments(c) and complete(c):
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
    ap.add_argument("--out", default=str(H / "gemma4_pretrain/data/dfin_v3"))
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
        upsample=upsample, heldout_excluded=n_held, seed=42, version="v3-excise-complete-tables", normalization="NFC + fullwidth alnum→halfwidth + special spaces→space + zero-width removed + fullwidth digit separators in numbers→ASCII (no NFKC: keeps CJK punctuation)",
        filter="drop chunk if >30% non-table lines are bare numbers or mean line len < 8; relayout tables replace text-layer annual",
    )
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
