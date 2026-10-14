"""Where does a carried state go bad? Val CE per 128K-token window over a long carried stream, plus state norms."""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TinyLM  # noqa
run, data = sys.argv[1], sys.argv[2]
n_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 2_097_152
ck = torch.load(os.path.join(run, "ckpt.pt"), map_location="cuda", weights_only=False)
m = TinyLM(50304, ck["args"]["d_model"], ck["args"]["layout"], ck["cfg"]).cuda().eval(); m.load_state_dict(ck["model"])
val = np.memmap(os.path.join(data, "val.bin"), dtype=np.uint16, mode="r")
st, T, win, out, losses = None, 2048, 131072, [], []
with torch.no_grad():
    for s in range(0, n_tok, T):
        x = torch.from_numpy(val[s:s + T + 1].astype(np.int64)).cuda()[None]
        with torch.autocast("cuda", torch.bfloat16):
            lg, st = m(x[:, :-1], st)
        losses.append(F.cross_entropy(lg.float()[0], x[0, 1:]).item())
        if (s + T) % win == 0:
            norms = [round(float(z[0].float().norm()), 2) for z in st if z is not None and torch.is_tensor(z[0])][:8]
            out.append({"tokens": s + T, "ce": round(float(np.mean(losses)), 3), "state_norms": norms}); losses = []
            print(json.dumps(out[-1]), flush=True)
json.dump(out, open(os.path.join(run, "stream_drift.json"), "w"), indent=1)
