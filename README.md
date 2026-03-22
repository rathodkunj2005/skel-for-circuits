# Graph Skeletonization for Attribution Graphs

This project simplifies attribution graphs in two phases:
1. **Prune** low-importance nodes.
2. **Group** the remaining nodes into **supernodes**, using either an embedding-based Mapper pipeline or an LLM-based grouping pipeline.

The resulting skeleton is saved back into the original graph format via `qParams` (pinned nodes + supernodes), so downstream visualization tools can render simplified graphs without losing original node metadata.

---

## Quick Start (Ollama — local, no API key needed)

This is the recommended path for teammates. All LLM calls run locally via [Ollama](https://ollama.com/).

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server (keep this running in a terminal):

```bash
ollama serve
```

Pull a model (gemma3:4b is small and fast; swap for any model you like):

```bash
ollama pull gemma3:4b
```

Check available models at any time:

```bash
ollama list
```

### 2. Clone & set up Python environment

```bash
git clone https://github.com/rathodkunj2005/skel-for-circuits.git
cd skel-for-circuits

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# .env already has USE_OLLAMA=true and OLLAMA_MODEL=gemma3:4b — edit if needed
```

If you use a different Ollama model, update `OLLAMA_MODEL` in `.env`:

```
OLLAMA_MODEL=llama3.2:3b
```

### 4. Run on a sample graph

```bash
USE_OLLAMA=true python3 -m experiments.run_experiment --graph data/sample_graphs/Gemma/gemma-fact-dallas-austin.json --grouping llm --open_source_grouping --open_source_labeling
```

Output lands in `data/outputs/`. That's it — no API key, no GPU required for small models.

---

## Other Run Modes

### Embedding-based grouping (no LLM for grouping)

```bash
USE_OLLAMA=true python3 -m experiments.run_experiment --graph data/sample_graphs/Gemma/gemma-fact-dallas-austin.json --grouping embedding
```

### Run on a whole directory of graphs

```bash
USE_OLLAMA=true python3 -m experiments.run_experiment --graph_dir data/sample_graphs/Gemma/ --grouping llm --open_source_grouping --open_source_labeling
```

### Use OpenAI instead of Ollama (requires API key)

```bash
export OPENAI_API_KEY=sk-...
python3 -m experiments.run_experiment --graph data/sample_graphs/Gemma/gemma-fact-dallas-austin.json --grouping llm
```

### Build a graph from a prompt (requires `circuit_tracer`)

The `external/Faithfulness` git submodule provides `circuit_tracer`. It requires SSH access to its upstream repo. If you have access:

```bash
git submodule update --init --recursive && pip install -e external/Faithfulness
```

Then:

```bash
USE_OLLAMA=true python3 -m experiments.run_experiment --prompt "The capital of France is"
```

---

## Model Selection

| Scenario | Recommended model | VRAM |
|---|---|---|
| Fast iteration / testing | `gemma3:4b` | ~4 GB |
| Better quality, still local | `llama3.2:3b` or `mistral:7b` | 4–8 GB |
| Best local quality | `llama3.3:70b` (Q4) | ~40 GB |
| Cloud / no local GPU | OpenAI (set `USE_OLLAMA=false`) | — |

Change the model any time by setting `OLLAMA_MODEL=<name>` in `.env` or as an env var prefix on the command.

---

## All CLI Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--graph` | str | — | Path to a single input graph JSON |
| `--graph_dir` | str | — | Path to a directory of graph JSON files |
| `--prompt` | str | — | Build attribution graph from this prompt (requires `circuit_tracer`) |
| `--grouping` | `embedding`\|`llm` | `embedding` | Grouping method: PCA-based Mapper or LLM-based clustering |
| `--open_source_grouping` | flag | off | Use local model for LLM-based **grouping** instead of OpenAI |
| `--open_source_labeling` | flag | off | Use local model for **auto-interpretation** and **umbrella terms** instead of OpenAI |
| `--n_groups_hint` | int | 0 | Soft hint for number of groups when `--grouping llm` (0 = no hint) |
| `--desired_logit_prob` | float | 0.95 | Target cumulative logit probability (prompt mode) |
| `--max_n_logits` | int | 10 | Max logit nodes (prompt mode) |
| `--max_feature_nodes` | int | 8192 | Max feature nodes (prompt mode) |
| `--graph_batch_size` | int | 256 | Batch size for attribution graph construction (prompt mode) |
| `--node_threshold` | float | 0.5 | Node pruning threshold (prompt mode) |
| `--edge_threshold` | float | 0.8 | Edge pruning threshold (prompt mode) |

Exactly one of `--graph`, `--graph_dir`, or `--prompt` must be provided.

---

## How It Works

### 1) Pruning (node reduction)

Implemented in `mapper/utils.py` and invoked by both pipeline variants via `prune_graph`:
- Removes MLP reconstruction error nodes.
- Scores remaining nodes with **TopImpactScorer**.
- Removes nodes with non-positive scores.

### 2) Supernode construction (grouping)

Two grouping pipelines are available, selected via `--grouping`:

#### `embedding` (default) — `MapperPipeline`

Groups nodes using a Mapper-style cover over a 1D lens:
- **Cover:** Overlapping intervals built from `TopImpactScorer` scores.
- **Similarity lens:** `SupernodeLens` (sentence-transformer embeddings, PCA-reduced to 1D) clusters nodes *within* each interval.
- **Clustering:** Complete-link thresholding groups nodes whose semantic values lie within a fixed distance.

#### `llm` — `LLMGroupingPipeline`

Groups nodes by asking an LLM to form semantically coherent clusters in a single call:
- Sends all node labels (indexed) to the LLM with a structured prompt.
- LLM returns group assignments and a concise label for each cluster.
- Accepts `--n_groups_hint` to softly guide the number of groups.

### 3) LLM-Assisted Labeling

LLMs are used in **two separate roles**, each independently switchable:

| Role | Triggered by | Controlled by |
|---|---|---|
| **Auto-interpretation** | Empty/missing node labels | `--open_source_labeling` |
| **Umbrella term** | Multi-node supernodes without a label | `--open_source_labeling` |
| **LLM grouping** | `--grouping llm` | `--open_source_grouping` |

When `USE_OLLAMA=true`, all three roles route through Ollama regardless of the `--open_source_*` flags.

### LLM Routing Priority

```
USE_OLLAMA=true  →  Ollama (local, any model via OLLAMA_MODEL)
USE_OLLAMA=false + --open_source_*  →  HuggingFace pipeline (local, OPEN_SOURCE_MODEL)
USE_OLLAMA=false (default)  →  OpenAI API (OPENAI_API_KEY required)
```

### TopImpactScorer

Scores every node by causal importance for the predicted output token:
1. Identifies sink nodes (top logit + high-probability alternatives).
2. Propagates backwards through the graph computing signed path sums.
3. Combines importance, competition, suppress-alt, gateway, and dominator sub-scores.
4. Applies node-type priors and a linear weighting formula.
5. Keeps nodes above `min_importance_for_selection = 0.035`.

---

## Project Structure

```
skel-for-circuits/
├── data/
│   ├── attr_graph.py          # Graph IO, format helpers (lazy circuit_tracer import)
│   ├── supernode_label.py     # LLM labeling — Ollama / OpenAI / HuggingFace backends
│   ├── sample_graphs/         # Example input graphs (Gemma, Llama, Qwen)
│   └── outputs/               # Generated skeleton JSONs land here
├── experiments/
│   └── run_experiment.py      # CLI entry point
├── lenses/
│   └── supernode.py           # SupernodeLens (sentence-transformer + PCA)
├── mapper/
│   ├── utils.py               # Pruning
│   ├── top_impact.py          # TopImpactScorer
│   └── llm_grouping.py        # LLMGroupingPipeline
├── external/
│   └── Faithfulness/          # circuit_tracer submodule (prompt mode only)
├── requirements.txt
├── .env.example               # Copy to .env and fill in
└── README.md
```

---

## Output Format

Each output JSON contains the original graph plus a `qParams` section:

```json
{
  "nodes": [...],
  "links": [...],
  "qParams": {
    "pinnedIds": ["node_id_1", "node_id_2", ...],
    "supernodes": [
      {
        "label": "clause-final punctuation prediction",
        "nodes": ["node_id_3", "node_id_4"]
      }
    ]
  }
}
```

Visualization tools can read `qParams` to overlay the skeleton without losing original node metadata.

---

## Input Format

Each input graph JSON must include:
- `nodes`: list of node objects with `node_id`, `label`/`clerp`, `feature_type`, `layer`, `ctx_idx`, `token_prob`, etc.
- `links`: list of directed edges with `source`, `target`, `weight`.
- `metadata`: prompt and run metadata (optional but used by the pipeline).

---

## Troubleshooting

**`ollama: command not found`** — Install Ollama from https://ollama.com/download.

**`Connection refused` on port 11434** — Run `ollama serve` in a separate terminal.

**Model not found** — Run `ollama pull <model>` first (e.g. `ollama pull gemma3:4b`).

**`ModuleNotFoundError: circuit_tracer`** — This only matters for `--prompt` mode. For JSON-graph mode it is safely skipped. If you need it, run `git submodule update --init --recursive && pip install -e external/Faithfulness`.

**Slow first run** — The sentence-transformer model (`all-MiniLM-L6-v2`) is downloaded on first use (~80 MB). Subsequent runs use the cache.

---

## References

Rosen, P., Hajij, M., & Wang, B. (2023, October). Homology-preserving multi-scale graph skeletonization using mapper on graphs. In *2023 Topological Data Analysis and Visualization (TopoInVis)* (pp. 10–20). IEEE.
