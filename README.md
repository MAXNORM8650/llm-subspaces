# LLM Subspaces

Extract the **knowledge subspaces** of any Hugging Face LLM and watch, **per generated token**, which subspaces the model uses — live, in a multi-turn chat.

For every target linear `W = UΣVᵀ`, we keep the top‑k right singular vectors `V_k` (k = effective rank at an energy threshold) as that layer's knowledge directions. During generation we hook the target linears and score each direction by `|σ_j · (v_jᵀ x)|` — "how much did this token use this subspace?"

## Features
- **Per-token attribution** — one teacher-forced forward captures which subspaces fire at *every* generated token (position `t` = the computation that predicts token `t+1`).
- **Multi-turn chat** — conversation history is kept; each turn is attributed.
- **Firing grid** — modules × layers; ▶ play or scrub the token timeline to watch it light up (query-specific, baseline-subtracted).
- **Leading-node trajectory** — connects the top-firing cell of each token to the next, drawing the path the dominant subspace traces through the network.
- **Click-to-drill** — top subspace directions for any cell at the selected token.
- **Sonification** — attention (sine) vs MLP (sawtooth), pitched by center-of-mass firing layer, volume by activity.

## Usage
```bash
pip install -r requirements.txt

# 1. build the atlas once (streamed SVD, works for any model size)
python subspace_atlas.py extract --model Qwen/Qwen2.5-3B-Instruct --out atlas_qwen3b --energy 0.95

# 2. serve the live app
python app_live.py --model Qwen/Qwen2.5-3B-Instruct --atlas atlas_qwen3b --port 8000 --web web --device cuda
# open http://localhost:8000
```

## Files
- `subspace_atlas.py` — streamed truncated-SVD atlas extraction + `SubspaceAttributor` (forward-hook attribution; per-token mode).
- `app_live.py` — dependency-free stdlib server: `/chat` (generate + per-token flow), `/drill`, `/reset`.
- `web/index.html` — the UI (firing grid, token timeline, leading-node path, drill panel, chat, sonification).

## Findings (Qwen2.5-3B)
- Knowledge subspaces are **high-rank** (r@95% ≈ 700–1700 of 2048); attention is more compressible than MLP.
- The **top-energy subspaces are general** — math/code/chat queries reuse nearly the same ones; per-query specialization shows up after baseline subtraction.

Notes: attribution hooks are gated to run only during the attribution pass, not during generation (fast decode). The atlas (`atlas_*/`, `*.pt`) is large and git-ignored — rebuild it with `extract`. MIT licensed.
