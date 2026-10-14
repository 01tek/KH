"""Control for the Mamba-1 delta-needle result (O1_RECALL_REPORT.md: recall 1.00 to 3200 tok).

Same eval protocol as deep_recall_eval.py, three conditions on identical prompts:
  bare      frozen Mamba-130m, no HAIS at all
  hais      hais_m3_delta_needle_best.pt (as reported)
  hais_nowrite  same checkpoint, but beta forced to 0 => the S-matrix is never written
Run with the env the checkpoint was built in:
  cd hais_engine && venv_torch27/bin/python poc_m3/verify_delta_needle.py
"""
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deep_recall_eval as E  # noqa: E402
from hais_m3_delta import inject_hais_m3_delta  # noqa: E402

TRIALS = int(os.environ.get("TRIALS", 30))
LENGTHS = [100, 400, 800, 1600, 3200]
DEPTHS = [0.0, 0.5, 0.95]


def load(cond, tok):
    m = AutoModelForCausalLM.from_pretrained(E.MODEL_ID, torch_dtype=E.DTYPE, device_map=E.DEVICE)
    if cond == "bare":
        m.reset_all_memory = lambda: None
    else:
        m = inject_hais_m3_delta(m)
        m.load_state_dict(torch.load(E.CKPT, map_location=E.DEVICE), strict=False)
        if cond == "hais_nowrite":
            for blk in m.backbone.layers:
                blk.hais_beta_proj.bias.data.fill_(-1e4)  # sigmoid -> 0: no writes, S stays 0
    return m.eval()


def run(model, tok, seed):
    random.seed(seed)
    out = {}
    for L in LENGTHS:
        hits = sum(E.run_trial(model, tok, E.sample_facts(1), [0.0], L, 0) for _ in range(TRIALS))
        out[f"len{L}"] = hits / TRIALS
    for d in DEPTHS:
        hits = sum(E.run_trial(model, tok, E.sample_facts(1), [d], 800, 0) for _ in range(TRIALS))
        out[f"depth{d}"] = hits / TRIALS
    return out


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tok = AutoTokenizer.from_pretrained(E.MODEL_ID)
    tok.pad_token = tok.eos_token
    res = {}
    for cond in ["bare", "hais", "hais_nowrite"]:
        model = load(cond, tok)
        res[cond] = run(model, tok, seed=1234)  # identical prompts across conditions
        print(cond, json.dumps(res[cond]), flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump({"trials_per_cell": TRIALS, **res}, open("poc_m3/verify_delta_needle.json", "w"), indent=2)


if __name__ == "__main__":
    main()
