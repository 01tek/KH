#!/usr/bin/env bash
# One-shot setup for a fresh GPU node (Hopper). Installs the verified stack, runs the 3 correctness gates,
# tokenises FineWeb-Edu. Usage: rsync poc_m3 to ~/hais_poc/poc_m3 on the node, then: bash ~/hais_poc/poc_m3/node_setup.sh [shards]
set -e
export PATH=$HOME/.local/bin:$PATH
export TOKENIZERS_PARALLELISM=false  # many prep workers x Rayon threads oversubscribed 160 cores and starved training 4x
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1)
cd ~/hais_poc
[ -d env ] || uv venv env --python 3.12 --python-preference only-managed -q
uv pip install -q --python env/bin/python torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python env/bin/python flash-linear-attention==0.5.2 "transformers>=4.45,<5" "huggingface-hub>=0.34,<1.0" einops packaging ninja numpy pytest py-spy pyarrow matplotlib
uv pip install -q --python env/bin/python triton==3.7.1   # Hopper: fla gated chunk bwd is wrong on triton 3.4-3.7.0 (fla #640)
uv pip install -q --python env/bin/python --no-deps "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
SHA=e9594ce1c732d97440f0332fdc43170a2294dbfa; SP=env/lib/python3.12/site-packages
curl -sfL https://raw.githubusercontent.com/state-spaces/mamba/$SHA/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py -o $SP/mamba_ssm/ops/triton/mamba3/mamba3_siso_fwd.py
curl -sfL https://raw.githubusercontent.com/state-spaces/mamba/$SHA/tests/ops/triton/test_mamba3_siso.py -o test_mamba3_siso.py
env/bin/python -m pytest -x -q -p no:warnings test_mamba3_siso.py | tail -1
cd poc_m3 && ../env/bin/python test_gdn_grad.py | tail -1 && ../env/bin/python test_hx_stream.py --null_sink | tail -1
cd lm
# 2B tokens first (enough for the phase-1 sweep, ready in minutes), then the full sample-10BT (~10B tokens) for phase 2
../../env/bin/python prep_data.py --out ~/data/fwe_2b --shards 3 --train_tokens 2000000000 --val_tokens 20000000 --workers $(nproc)
echo DATA_2B_READY
../../env/bin/python prep_data.py --out ~/data/fwe --shards ${1:-14} --train_tokens 20000000000 --val_tokens 20000000 --workers $(nproc)
echo NODE_READY
