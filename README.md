# LLM Subspace Atlas

**Extract the knowledge subspaces of any LLM's weights, and see — for any question — which subspaces the information flows through, layer by layer.**

Every linear weight `W = UΣVᵀ` has a set of *knowledge directions* (its right singular vectors `V`). This toolkit (1) **extracts** those subspaces at scale for any HuggingFace model, and (2) **attributes**, for a given query, how strongly each subspace is used at each layer — rendered as an interactive **Sankey flow** from input to output.

![demo](web/screenshot.png)

## Why
Fine-tuning and Mixture-of-Experts adapters implicitly reuse these subspaces. Making them explicit lets you:
- **initialize experts on real knowledge subspaces** instead of random directions,
- **route** tokens by which subspace they need,
- and **interpret** which knowledge a query actually engages.

## Install
```bash
pip install -r requirements.txt
```

## 1. Extract the atlas (scales to any model — streams one weight at a time)
```bash
python subspace_atlas.py extract --model Qwen/Qwen2.5-3B-Instruct --out atlas_qwen3b --energy 0.95
```
Saves `{module: V_k, σ_k, eff_rank}` per target linear (k = per-weight rank at 95% energy).

## 2a. Live — type any question, see its flow
```bash
python app.py --model Qwen/Qwen2.5-3B-Instruct --atlas atlas_qwen3b
# open http://localhost:8000, type a question
```

## 2b. Static site (precomputed queries, GitHub-Pages ready)
```bash
python export_flow.py --model Qwen/Qwen2.5-3B-Instruct --atlas atlas_qwen3b \
    --query "Solve for x: 3x+7=22" --tag math --out web/data
# repeat for other queries, then:
cd web && python -m http.server 8080   # or push web/ to GitHub Pages
```

## How the attribution works
For input activation `x` into a weight, usage of subspace direction `j` is
`usage_j = |σ_j · (v_jᵀ x)|`, averaged over tokens. The Sankey ribbon for each
`(layer, module)` is its total usage; hover shows the top subspace directions activated.

## What it shows today vs. next
- **v1 (now):** per-layer subspace **usage** — which subspaces the query touches on its way to the output.
- **v2 (roadmap):** *causal* cross-subspace routing (which subspace in layer L feeds which in L+1) via path patching; discriminative-subspace analysis (which subspaces separate domains); subspace-initialized MoE experts.

## Findings (Qwen2.5-3B)
- Knowledge subspaces are **high-rank** (r@95% ≈ 700–1700 of 2048); attention is more compressible than MLP.
- The **top-energy subspaces are general** — math/code/chat queries use nearly the same ones (cosine ≈ 0.99). Domain specialization, if any, lives in the **discriminative** (not top-energy) directions.

## Layout
```
subspace_atlas.py   extract + attribute (library + CLI)
export_flow.py      query -> Sankey JSON
app.py              live Flask server (any question)
web/                static viz (D3 sankey) + precomputed data/
```
MIT licensed.
