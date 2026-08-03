"""LLM Subspace Atlas — scalable knowledge-subspace extraction + query attribution for ANY HF LLM.

extract:   per target linear W=UΣVᵀ, keep top-k right singular vectors V_k (k = per-weight
           effective rank at an energy threshold). Streamed one weight at a time -> works for
           any model size. Saves a reusable atlas to disk.
attribute: forward hooks capture input activations x; usage_j = |σ_j·(v_jᵀ x)| per subspace
           direction -> "which knowledge subspaces did this query use?" (per module + aggregated).

Usage:
  python subspace_atlas.py extract   --model Qwen/Qwen2.5-3B-Instruct --out atlas_qwen3b --energy 0.95
  python subspace_atlas.py attribute --model Qwen/Qwen2.5-3B-Instruct --atlas atlas_qwen3b --demo
"""
import argparse
import glob
import json
import os

import torch

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _iter_target_linears(model):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in TARGETS:
            yield name, mod


# ---------------------------------------------------------------- extract
def extract(model_name, out_dir, energy=0.95, kmax=2048, device="cuda", dtype=torch.float16):
    from transformers import AutoModelForCausalLM
    os.makedirs(out_dir, exist_ok=True)
    print(f"[atlas] loading {model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    index = {}
    for name, mod in _iter_target_linears(model):
        W = mod.weight.data.float().to(device)                 # (out, in) — one weight at a time
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)    # V = Vh.T : (in, k)
        e = S.pow(2); cum = torch.cumsum(e, 0) / e.sum()
        k = min(int((cum < energy).sum()) + 1, kmax)
        Vk = Vh[:k].T.contiguous().to(dtype).cpu()             # (in, k) right singular vectors
        Sk = S[:k].contiguous().to(dtype).cpu()
        torch.save({"V": Vk, "sigma": Sk, "eff_rank": k, "shape": tuple(W.shape)},
                   os.path.join(out_dir, name.replace(".", "__") + ".pt"))
        index[name] = {"eff_rank": k, "shape": list(W.shape), "in": W.shape[1]}
        del W, U, S, Vh; torch.cuda.empty_cache()
    json.dump({"model": model_name, "energy": energy, "modules": index},
              open(os.path.join(out_dir, "index.json"), "w"), indent=2)
    tot = sum(v["eff_rank"] for v in index.values())
    print(f"[atlas] extracted {len(index)} modules, total eff-rank dims = {tot} -> {out_dir}", flush=True)


# ---------------------------------------------------------------- attribute
class SubspaceAttributor:
    def __init__(self, model, atlas_dir, device="cuda"):
        self.model = model
        self.device = device
        self.atlas = {}
        for f in glob.glob(os.path.join(atlas_dir, "*__*.pt")):
            name = os.path.basename(f)[:-3].replace("__", ".")
            d = torch.load(f, map_location=device)
            self.atlas[name] = (d["V"].float(), d["sigma"].float())     # V:(in,k) sigma:(k,)
        self._hooks, self._usage = [], {}
        name2mod = dict(model.named_modules())
        for name, (V, sig) in self.atlas.items():
            mod = name2mod[name]
            mod._atlas_name = name
            self._hooks.append(mod.register_forward_pre_hook(self._hook))

    def _hook(self, mod, args):
        x = args[0]                                              # (B, T, in)
        V, sig = self.atlas[mod._atlas_name]
        proj = torch.einsum("bti,ik->btk", x.float(), V)        # (B,T,k) coeff on each direction
        u = (proj.abs() * sig).mean(dim=(0, 1))                 # (k,) mean |σ_j·(v_jᵀx)| over tokens
        self._usage[mod._atlas_name] = u.detach().cpu()

    @torch.no_grad()
    def attribute(self, input_ids):
        self._usage = {}
        self.model(input_ids.to(self.device))
        return dict(self._usage)                                 # {module: usage_vector(k)}

    def close(self):
        for h in self._hooks:
            h.remove()


def _demo(model_name, atlas_dir, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    attr = SubspaceAttributor(model, atlas_dir, device)
    queries = {
        "math": "Solve for x: 3x + 7 = 22. Show your steps.",
        "code": "Write a Python function that returns the nth Fibonacci number.",
        "chat": "What's a good way to stay motivated on a long project?",
    }
    prof = {}
    for tag, q in queries.items():
        ids = torch.tensor([tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True)])
        usage = attr.attribute(ids)
        # concatenate a global fingerprint (normalized top-dims per module)
        vec = torch.cat([u / (u.sum() + 1e-9) for u in usage.values()])
        prof[tag] = vec
        # report a representative module's top subspace directions
        g = usage["model.layers.18.mlp.gate_proj"]
        top = g.topk(5).indices.tolist()
        print(f"[{tag:>4}] gate_proj L18 top-5 subspace dirs = {top}  (total usage {g.sum():.1f})")
    print("\n=== cross-query subspace-usage cosine similarity (lower = more distinct) ===")
    tags = list(prof)
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            c = torch.nn.functional.cosine_similarity(prof[tags[i]], prof[tags[j]], dim=0).item()
            print(f"  {tags[i]:>4} vs {tags[j]:>4}: cos = {c:.3f}")
    attr.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--model", required=True); e.add_argument("--out", required=True)
    e.add_argument("--energy", type=float, default=0.95); e.add_argument("--kmax", type=int, default=2048)
    a = sub.add_parser("attribute"); a.add_argument("--model", required=True); a.add_argument("--atlas", required=True)
    a.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.model, args.out, args.energy, args.kmax)
    elif args.cmd == "attribute" and args.demo:
        _demo(args.model, args.atlas)
