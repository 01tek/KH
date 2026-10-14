"""HAIS-X correctness: with tau=0 (store keeps everything) chunked streaming must equal a full pass."""
import torch

from model import TinyLM
from tasks import Vocab, fact_stream

torch.manual_seed(0)
V = Vocab()
import sys
sink = "--null_sink" in sys.argv
m = TinyLM(V.size, 256, "hais,hx", {"tau": 0.0, "store_cap": 1 << 20, "null_sink": sink}).cuda().eval()
ids, _, _ = fact_stream(2, 1024, 8, vocab=V)
with torch.no_grad():
    full, _ = m(ids)
    st, outs = None, []
    for s in range(0, 1024, 256):
        o, st = m(ids[:, s:s + 256], st)
        outs.append(o)
    ch = torch.cat(outs, 1)
d = (full - ch).abs().max().item()
print(f"hx stream max|diff| fp32 = {d:.2e} (logit scale {full.abs().max().item():.2f}); store entries = "
      f"{m.blocks[1].mixer.store_size(st[1])}")
assert d < 1e-2, "HAIS-X streaming mismatch"
print("HX_STREAM_OK")
