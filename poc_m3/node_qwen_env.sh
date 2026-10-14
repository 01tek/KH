#!/bin/bash
# Qwen3.5 environment on the B200 node (mirrors H200 #1 env_qwen) + Qwen3.5 weights in the HF cache.
cd ~/hais_poc
command -v uv >/dev/null || [ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -x env_qwen/bin/python ] || uv venv env_qwen --python 3.12 --python-preference only-managed -q
uv pip install -q --python env_qwen/bin/python torch==2.13.0 transformers==5.19.0 flash-linear-attention==0.5.2 \
  fla-core==0.5.2 triton==3.7.1 tokenizers==0.23.3 huggingface-hub==2.2.0 einops numpy accelerate pyarrow
env_qwen/bin/python -c "import torch, transformers, fla; print('torch', torch.__version__, torch.cuda.is_available(), 'tf', transformers.__version__, 'fla', fla.__version__)"
for m in Qwen/Qwen3.5-4B Qwen/Qwen3.5-0.8B; do
  env_qwen/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$m', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','tokenizer*','*.model']))"
done
echo QWEN_ENV_DONE
