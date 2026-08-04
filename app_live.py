"""Live LLM Subspace Atlas — multi-turn chat with PER-TOKEN subspace attribution.

Dependency-free (stdlib http.server). For each assistant turn it generates the full
response, then does ONE teacher-forced forward to capture which knowledge subspaces
fire at EVERY generated token. The UI (web/index.html) plays/scrubs through tokens,
lighting the per-layer/per-module firing grid, with click-to-drill on any cell+token.

  python app_live.py --model Qwen/Qwen2.5-3B-Instruct --atlas atlas_qwen3b --port 8000 --web web --device cuda
"""
import argparse
import http.server
import json
import re
import urllib.parse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subspace_atlas import SubspaceAttributor

FAM = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn",
       "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
S = {}


def _cell(name):
    m = re.search(r"layers\.(\d+)\.\w+\.(\w+)", name)
    return (int(m.group(1)), m.group(2)) if m else (None, None)


@torch.no_grad()
def chat_turn(user_msg, max_new=200):
    """Append user msg -> generate assistant reply -> per-token subspace attribution."""
    S["messages"].append({"role": "user", "content": user_msg})
    try:                                                      # disable thinking mode if the template supports it
        toks = S["tok"].apply_chat_template(S["messages"], add_generation_prompt=True, tokenize=True, enable_thinking=False)
    except TypeError:
        toks = S["tok"].apply_chat_template(S["messages"], add_generation_prompt=True, tokenize=True)
    ids = torch.tensor([toks]).to(S["model"].device)
    plen = ids.shape[1]
    gen = S["model"].generate(ids, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=(S["tok"].eos_token_id or 0))
    seq = gen[0]                                              # (plen + G,)
    gtoks = seq[plen:]
    answer = S["tok"].decode(gtoks, skip_special_tokens=True).strip()
    S["messages"].append({"role": "assistant", "content": answer})
    G = int(gtoks.shape[0])
    tok_strs = [S["tok"].decode([int(t)]) for t in gtoks]     # per-generated-token text

    # per-position attribution over the full sequence; position t predicts token t+1,
    # so generated token j (at plen+j) is predicted at position off+j, off = plen-1.
    per = S["attr"].attribute_tokens(seq.unsqueeze(0))        # {name: (sum(T,), idx(T,16), val(T,16))}
    off = max(0, plen - 1)
    base = S.get("baseline_sum", {})
    cells, grid, active, drill = [], [], [], {"idx": {}, "val": {}}
    order = sorted(per, key=lambda n: (_cell(n)[0] if _cell(n)[0] is not None else 99, _cell(n)[1]))
    for name in order:
        L, proj = _cell(name)
        if L is None:
            continue
        cid = f"L{L}.{proj}"
        cells.append(cid)
        s, idx, val = per[name]                               # (T,), (T,16), (T,16)
        b = float(base.get(name, 0.0))
        uu = (s[off:off + G] - b).clamp(min=0)                # (G,) query/token-specific usage
        grid.append([round(float(x), 3) for x in uu.tolist()])
        drill["idx"][cid] = idx[off:off + G].tolist()         # (G,16)
        drill["val"][cid] = [[round(float(x), 2) for x in row] for row in val[off:off + G].tolist()]
    # active modules per generated token (how many of 252 cells fire above baseline)
    gt = torch.tensor(grid)                                   # (cells, G)
    active = [int((gt[:, j] > 0).sum()) for j in range(G)] if G else []
    fam = [FAM.get(c.split(".")[1], "other") for c in cells]
    S["last"] = {"tokens": tok_strs, "cells": cells, "drill": drill}
    return {"answer": answer, "tokens": tok_strs, "cells": cells, "fam": fam,
            "grid": grid, "active": active, "n_cells": len(cells),
            "total_dirs": S["total_dirs"], "scale": S.get("scale", 1.0),
            "history": S["messages"]}


def drill(tok, cid):
    L = S.get("last", {})
    if cid not in L.get("drill", {}).get("idx", {}) or tok >= len(L["tokens"]):
        return {"error": "no data"}
    return {"token": L["tokens"][tok], "cell": cid,
            "dirs": L["drill"]["idx"][cid][tok], "vals": L["drill"]["val"][cid][tok]}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=S["web"], **k)

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/chat":
            msg = q.get("q", [""])[0].strip()
            mx = max(1, min(1024, int(q.get("maxn", ["200"])[0])))
            self._json(chat_turn(msg, mx) if msg else {"error": "empty"})
        elif u.path == "/drill":
            self._json(drill(int(q.get("tok", ["0"])[0]), q.get("cell", [""])[0]))
        elif u.path == "/reset":
            S["messages"] = []
            self._json({"ok": True})
        else:
            super().do_GET()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--atlas", required=True)
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--web", default="web")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    S["web"] = a.web; S["messages"] = []
    print("loading model + atlas ...", flush=True)
    S["tok"] = AutoTokenizer.from_pretrained(a.model)
    _dt = torch.float32 if a.device == "cpu" else torch.bfloat16
    S["model"] = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=_dt).to(a.device).eval()
    S["attr"] = SubspaceAttributor(S["model"], a.atlas, a.device)
    idx = json.load(open(f"{a.atlas}/index.json"))
    S["total_dirs"] = sum(v["eff_rank"] for v in idx["modules"].values())
    print("computing baseline (general subspace usage) ...", flush=True)
    _bp = ["Solve 2 plus 2.", "Write a Python loop.", "How are you today?", "Name a color."]
    _acc = {}
    with torch.no_grad():
        for p in _bp:
            _ids = torch.tensor([S["tok"].apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)]).to(S["model"].device)
            for k, v in S["attr"].attribute(_ids).items():
                _acc[k] = _acc.get(k, v * 0) + v
    S["baseline"] = {k: v / len(_bp) for k, v in _acc.items()}
    S["baseline_sum"] = {k: float(v.sum()) for k, v in S["baseline"].items()}   # per-module baseline (for per-token subtraction)
    S["scale"] = max((float(v.sum()) for v in S["baseline"].values()), default=1.0) * 2.0
    print(f"baseline ready ({len(S['baseline'])} modules), total dirs={S['total_dirs']} — per-token attribution live", flush=True)
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"READY: serving on 0.0.0.0:{a.port} (host {__import__('socket').gethostname()})", flush=True)
    srv.serve_forever()
