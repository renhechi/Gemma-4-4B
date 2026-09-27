# LGDSRFPA × Gemma-4-E4B

Learning-Goal-Driven Structural Regulation Further Pretraining (LGDSRFPA) — the
Gemma-3-TAIDE-12B design of the PhD proposal re-implemented on `google/gemma-4-E4B-it`
(last-layer Gated FFN, d=2560, p=10240, D* = 20 × 78.64M = 1.573B tokens).

- Full experiment procedure (E1–E5): [`docs/EXPERIMENT_PROTOCOL_E4B.md`](docs/EXPERIMENT_PROTOCOL_E4B.md)
- Trainer (E1/E2/E3): `scripts/train_lgdsrfpa.py`, structure-adjustable FFN: `src/lgdsrfpa4/struct_ffn.py`
- Corpus build: `scripts/build_dfin.py` · analysis: `scripts/analyze_runs.py` · final eval: `scripts/eval_final.py`
- Run metrics: `runs/<experiment>/metrics.jsonl` (checkpoints / model / data are not in the repo)

Environment: ASUS Ascent GX10 (GB10, 128GB unified), conda `gemma3_env` (torch 2.11+cu128, transformers 5.8).
