"""Local live server: type ANY question -> see its subspace-usage flow in the browser.
Loads the model + atlas once, serves the web UI and a /flow?q=... endpoint.

  pip install -r requirements.txt
  python subspace_atlas.py extract --model Qwen/Qwen2.5-3B-Instruct --out atlas_qwen3b
  python app.py --model Qwen/Qwen2.5-3B-Instruct --atlas atlas_qwen3b
  # open http://localhost:8000
"""
import argparse
import re

import torch
from flask import Flask, jsonify, request, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer

from subspace_atlas import SubspaceAttributor

FAM = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn",
       "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
app = Flask(__name__, static_folder="web")
STATE = {}


def build_flow(usage, query, topd=8):
    nodes, per = {}, {}
    for name, u in usage.items():
        m = re.search(r"layers\.(\d+)\.\w+\.(\w+)", name)
        if not m:
            continue
        L, proj = int(m.group(1)), m.group(2)
        nid = f"L{L}.{proj}"
        nodes[nid] = {"id": nid, "layer": L, "module": proj, "family": FAM.get(proj, "other"),
                      "usage": round(float(u.sum()), 2),
                      "top_dirs": u.topk(min(topd, u.numel())).indices.tolist(), "eff_rank": int(u.numel())}
        per.setdefault(L, {})[proj] = float(u.sum())
    Ls = sorted(per); links = []
    for a, b in zip(Ls, Ls[1:]):
        for proj in per[a]:
            if proj in per[b]:
                links.append({"source": f"L{a}.{proj}", "target": f"L{b}.{proj}",
                              "value": round(min(per[a][proj], per[b][proj]), 2), "family": FAM.get(proj, "other")})
    return {"query": query, "nodes": list(nodes.values()), "links": links}


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/<path:p>")
def static_files(p):
    return send_from_directory("web", p)


@app.route("/flow")
def flow():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "empty query"}), 400
    tok, model, attr = STATE["tok"], STATE["model"], STATE["attr"]
    ids = torch.tensor([tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True)])
    return jsonify(build_flow(attr.attribute(ids), q))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--atlas", required=True)
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    STATE["tok"] = AutoTokenizer.from_pretrained(a.model)
    STATE["model"] = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
    STATE["attr"] = SubspaceAttributor(STATE["model"], a.atlas, a.device)
    print(f"serving http://localhost:{a.port}")
    app.run(host="0.0.0.0", port=a.port)
