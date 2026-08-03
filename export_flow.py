"""Export a query's subspace-usage 'flow' as Sankey JSON for the web viz.
Nodes = (layer, module); ribbon/usage = how strongly that module's knowledge subspace is
used for the query; each node carries its top subspace directions (hover detail).
Links = per-module channels flowing across consecutive layers (usage carried forward).

  python export_flow.py --model Qwen/Qwen2.5-3B-Instruct --atlas atlas_qwen3b \
      --query "Solve 3x+7=22" --tag math --out web/data
"""
import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from subspace_atlas import SubspaceAttributor

FAM = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn",
       "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


def export(model_name, atlas, query, tag, out_dir, topd=8, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    attr = SubspaceAttributor(model, atlas, device)
    ids = torch.tensor([tok.apply_chat_template(
        [{"role": "user", "content": query}], add_generation_prompt=True, tokenize=True)])
    usage = attr.attribute(ids)
    attr.close()

    nodes, per = {}, {}
    for name, u in usage.items():
        m = re.search(r"layers\.(\d+)\.\w+\.(\w+)", name)
        if not m:
            continue
        L, proj = int(m.group(1)), m.group(2)
        tot = float(u.sum())
        top = u.topk(min(topd, u.numel())).indices.tolist()
        nid = f"L{L}.{proj}"
        nodes[nid] = {"id": nid, "layer": L, "module": proj, "family": FAM.get(proj, "other"),
                      "usage": round(tot, 2), "top_dirs": top, "eff_rank": int(u.numel())}
        per.setdefault(L, {})[proj] = tot
    Ls = sorted(per)
    links = []
    for a, b in zip(Ls, Ls[1:]):
        for proj in per[a]:
            if proj in per[b]:
                links.append({"source": f"L{a}.{proj}", "target": f"L{b}.{proj}",
                              "value": round(min(per[a][proj], per[b][proj]), 2),
                              "family": FAM.get(proj, "other")})
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.json")
    json.dump({"query": query, "tag": tag, "model": model_name,
               "nodes": list(nodes.values()), "links": links}, open(path, "w"))
    print(f"wrote {path}: {len(nodes)} nodes, {len(links)} links")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--atlas", required=True)
    ap.add_argument("--query", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="web/data")
    a = ap.parse_args()
    export(a.model, a.atlas, a.query, a.tag, a.out)
