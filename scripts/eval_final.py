#!/usr/bin/env python3
"""ED §4.9 final evaluation: base Gemma-4-E4B-it vs LGDSRFPA checkpoint(s).

  1) PPL on FinKD held-out (zh 400 / en 400; excluded from D_fin by build_dfin.py)
  2) Numeric financial QA (qa_benchmark_350_zh, same scoring as gemma3 track)
  3) Expert-evaluation kit (run ONCE, base vs final model): blind A/B answers to
     eval/expert_questions.jsonl + rubric (術語正確性/法規語境一致性/回答完整性/幻覺程度/實務可用性)
     → docs/results/expert_kit/{blind_pairs.csv, answer_key.csv, rubric.md}

Usage:
  python3 scripts/eval_final.py --ckpt runs/E3-full/dose_1.00.pt [--ckpt runs/E1-full/dose_1.00.pt] [--expert-kit]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from lgdsrfpa4.struct_ffn import StructGatedFFN  # noqa: E402

H = Path.home()
BASE = ROOT / "models/gemma-4-E4B-it"
QA = H / "thesis_repo/results/qa_benchmark_350_zh.jsonl"
HELD = H / "gemma3_corpus_v2/ppl_heldout"
OUT = ROOT / "docs/results"
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def load(ckpt: str | None):
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(BASE)
    m = Gemma4ForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa")
    for a in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(m.model, a):
            setattr(m.model, a, None)
    m.to("cuda").eval()
    if ckpt:
        s = torch.load(ckpt, map_location="cuda", weights_only=False)
        layer = m.model.language_model.layers[s["layer"]]
        smlp = StructGatedFFN(layer.mlp)
        for n, t in s["mlp"].items():
            setattr(smlp, n, nn.Parameter(t.to("cuda")))
        layer.mlp = smlp
        print(f"[load] {ckpt}: layer {s['layer']} p={smlp.p} dose={s.get('dose')} tokens={s.get('tokens_seen')}")
    return m, tok


@torch.no_grad()
def ppl(m, tok, path, max_len=1024):
    nll = n = 0
    for line in open(path):
        ids = tok(json.loads(line)["text"], add_special_tokens=False, return_tensors="pt")["input_ids"][:, :max_len].cuda()
        if ids.shape[1] < 16:
            continue
        loss = m(input_ids=ids, labels=ids).loss.float().item()
        nll += loss * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return math.exp(nll / n)


def chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def generate(m, tok, prompts, max_new=40, B=8):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), B):
        enc = tok([chat(tok, p) for p in prompts[i:i + B]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def qa_eval(m, tok):
    qa = [json.loads(l) for l in open(QA)]
    preds = generate(m, tok, [f"{q['context']}\n\n問題：{q['question']}" for q in qa])
    hit = {"extract": [0, 0], "compute": [0, 0]}
    for q, p in zip(qa, preds):
        nums = _NUM.findall(p.replace(",", ""))
        v = float(nums[0]) if nums else None
        ok = v is not None and (abs(v - q["answer"]) / abs(q["answer"]) <= q["tol"] if abs(q["answer"]) > 1e-9 else abs(v) < q["tol"])
        hit[q["qtype"]][0] += ok
        hit[q["qtype"]][1] += 1
    tot = sum(h[0] for h in hit.values()) / len(qa)
    return {"qa_acc": round(tot, 4), **{f"qa_{k}": round(h[0] / h[1], 4) for k, h in hit.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--expert-kit", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res, expert = {}, {}
    qs = [json.loads(l) for l in open(ROOT / "eval/expert_questions.jsonl")] if args.expert_kit else []
    for ck in [None] + args.ckpt:
        name = "base" if ck is None else Path(ck).parent.name + "/" + Path(ck).stem
        m, tok = load(ck)
        r = {"ppl_zh": round(ppl(m, tok, HELD / "zh_heldout.jsonl"), 4),
             "ppl_en": round(ppl(m, tok, HELD / "en_heldout.jsonl"), 4)}
        if not args.skip_qa:
            r.update(qa_eval(m, tok))
        if qs and (ck is None or ck == args.ckpt[-1]):
            expert[name] = generate(m, tok, [q["question"] for q in qs], max_new=512, B=4)
        res[name] = r
        print(name, r, flush=True)
        del m
        torch.cuda.empty_cache()
    (OUT / "final_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))

    if qs and len(expert) == 2:
        kit = OUT / "expert_kit"
        kit.mkdir(exist_ok=True)
        (a_name, a), (b_name, b) = expert.items()
        rng = random.Random(42)
        with open(kit / "blind_pairs.csv", "w", newline="") as f1, open(kit / "answer_key.csv", "w", newline="") as f2:
            w1, w2 = csv.writer(f1), csv.writer(f2)
            dims = ["術語正確性", "法規語境一致性", "回答完整性", "幻覺程度(低=好)", "實務可用性"]
            w1.writerow(["id", "category", "question", "answer_A", "answer_B"] + [f"A_{d}" for d in dims] + [f"B_{d}" for d in dims] + ["偏好(A/B/平手)", "評語"])
            w2.writerow(["id", "A_is", "B_is"])
            for i, q in enumerate(qs):
                swap = rng.random() < 0.5
                x, y = (b[i], a[i]) if swap else (a[i], b[i])
                w1.writerow([q["id"], q["category"], q["question"], x, y] + [""] * 11)
                w2.writerow([q["id"], b_name if swap else a_name, a_name if swap else b_name])
        (kit / "rubric.md").write_text(RUBRIC)
        print(f"[kit] → {kit}")


RUBRIC = """# 金融專業人士評分說明(ED §4.9,僅施測一次)

比較對象:原始 Gemma-4-E4B-it vs 完成 D_fin 觀測流程之 LGDSRFPA 模型。兩個回答以 A/B 隨機匿名呈現,
請勿嘗試推測來源;answer_key.csv 由研究者保管,評分完成後才開啟。

每一構面 1–5 分(5 = 最佳):
| 構面 | 5 分 | 3 分 | 1 分 |
|---|---|---|---|
| 術語正確性 | 金融/會計/保險術語使用完全正確 | 少數術語不精確 | 術語錯誤或誤用 |
| 法規語境一致性 | 符合臺灣現行法規(證交法、銀行法、保險法、金管會規範)語境 | 大致正確但有模糊處 | 與法規矛盾或套用他國法規 |
| 回答完整性 | 涵蓋問題所有要點並有適當說明 | 回答主要問題但遺漏要點 | 答非所問或嚴重缺漏 |
| 幻覺程度(低=好) | 無捏造事實/數字/條文 | 有輕微不可驗證陳述 | 捏造條文、數字或機構 |
| 實務可用性 | 可直接作為實務參考 | 需修改後使用 | 不具實務價值 |

最後請填「偏好」(A / B / 平手)與自由評語。
"""

if __name__ == "__main__":
    main()
