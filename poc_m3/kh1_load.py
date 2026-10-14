"""Load KH-1 adapters from a training checkpoint (.pt) or a release folder (config.json + adapters.safetensors)."""
import json
import os

import torch


def load_adapters(path):
    """Returns (cfg, state_dict). `path` is a ckpt.pt from train_lm.py or a Hugging Face release folder."""
    if os.path.isdir(path):
        from safetensors.torch import load_file
        cfg = json.load(open(os.path.join(path, "config.json")))
        return cfg, load_file(os.path.join(path, "adapters.safetensors"))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["cfg"], ck["model"]
