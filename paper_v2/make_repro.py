"""Rebuild the exact training command line of every headline run from the "config" event its training log recorded.

  python paper_v2/make_repro.py > paper_v2/repro/commands.md
Each command is what train_lm.py was actually called with (argparse namespace as logged), so it reproduces the run
up to hardware nondeterminism. Run from poc_m3/lm/ with the data directories prepared by prep_data.py.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
RUNS = [  # (label, training log, hardware, train-time source)
    ("117M from scratch, far drills, GDN cap 256 (s0)", "b200_node/runs_s5/s5_hl256_far.train.log", "1 B200 (MPS-shared)"),
    ("117M from scratch, far drills, GDN cap 256 (s1)", "b200_node/runs_s5/s5_hl256_far_s1.train.log", "1 B200 (MPS-shared)"),
    ("117M from scratch, far drills, GDN cap 512 (s0)", "b200_node/runs_s5/s5_hl512_far.train.log", "1 B200 (MPS-shared)"),
    ("117M from scratch, far drills, GDN cap 512 (s1)", "b200_node/runs_s5/s5_hl512_far_s1.train.log", "1 B200 (MPS-shared)"),
    ("494M from scratch, far drills (v4)", "b200_node/runs_node/gdnhais_d1536_v4.train.log", "8x B200"),
    ("KH-1-350M stage 1 (pilot 2)", "h200_92/runs_lfm2/lfm2_350m_pilot2.train.log", "1 H200 (MPS-shared)"),
    ("KH-1-350M stage 2 (pilot 3)", "h200_92/runs_lfm2/lfm2_350m_pilot3.train.log", "1 H200 (MPS-shared)"),
    ("KH-1-350M stage 3 (pilot 4)", "h200_92/runs_lfm2/lfm2_350m_pilot4.train.log", "1 H200 (MPS-shared)"),
    ("KH-1-350M stage 4 (pilot 6, released)", "h200_92/runs_lfm2/lfm2_350m_pilot6.train.log", "1 H200 (MPS-shared)"),
    ("KH-1-1.2B v1", "b200_node/runs_kh1/kh1_1p2b.train.log", "8x B200"),
    ("KH-1-1.2B v2", "b200_node/runs_kh1/kh1_1p2b_t09.train.log", "8x B200"),
    ("KH-1-1.2B v4 (released)", "b200_node/runs_kh1/kh1_1p2b_v4.train.log", "8x B200"),
    ("KH-1-Q-4B stage 1", "b200_node/runs_qwen/q4b_s1.train.log", "8x B200"),
    ("KH-1-Q-4B stage 2 (released)", "b200_node/runs_qwen/q4b_s2.train.log", "8x B200"),
]
SKIP = {"world", "event", "n_params", "steps"}  # logged extras, not CLI flags
BOOL = {"compile", "lfm2_unfreeze", "no_think"}


def cli(cfg):
    parts = []
    for k, v in cfg.items():
        if k in SKIP or v is None or v is False:
            continue
        if v is True or k in BOOL:
            parts.append(f"--{k}")
            continue
        parts.append(f"--{k} {v}")
    return "python train_lm.py " + " \\\n  ".join(parts)


print("# Exact training commands (rebuilt from the logged configs)\n")
print("Paths in `--data`, `--out` and `--init_from` are those of the original machines; data directories come from "
      "`prep_data.py` (Appendix). Multi-GPU runs used `torchrun --standalone --nproc_per_node=8 train_lm.py ...` with the "
      "same flags; `--bs` is per GPU.\n")
for label, path, hw in RUNS:
    p = os.path.join(R, path)
    cfg = done = None
    for line in open(p):
        if line.startswith('{"event": "config"'):
            cfg = json.loads(line)
        elif line.startswith('{"event": "done"'):
            done = json.loads(line)
    if cfg is None:
        print(f"## {label}\nconfig event not found in {path}\n", file=sys.stderr)
        continue
    world = cfg.get("world", 1)
    t = done["train_time_s"] if done else None
    print(f"## {label}\n")
    print(f"- log: `results/{path}`; hardware: {hw}; world size {world}; "
          f"{cfg.get('n_params', 0) / 1e6:.0f}M total parameters; "
          f"{cfg['tokens'] / 1e6:.0f}M training tokens; wall clock {t / 60:.1f} min" if t else "")
    print("\n```bash\n" + cli({k: v for k, v in cfg.items()}) + "\n```\n")
