# Graph Skeletonization for Attribution Graphs

This project simplifies attribution graphs in two phases:
1) **Prune** low-importance nodes.
2) **Group** the remaining nodes into **supernodes**, using either an embedding-based Mapper pipeline or an LLM-based grouping pipeline.

The resulting skeleton is saved back into the original graph format via `qParams` (pinned nodes + supernodes), so downstream visualization tools can render simplified graphs without losing original node metadata.

## How It Works

### 1) Pruning (node reduction)
Pruning is implemented in [graph_skel/mapper/utils.py](graph_skel/mapper/utils.py) and is invoked by both pipeline variants via `prune_graph`.

Current behavior:
- Removes MLP reconstruction error nodes.
- Computes a score for every remaining node using **TopImpactScorer**.
- Removes nodes with non-positive scores.

### 2) Supernode construction (grouping)

Two grouping pipelines are available, selected via `--grouping`:

#### `embedding` (default) — `MapperPipeline`
Groups nodes using a Mapper-style cover over a 1D lens:
- **Cover:** Overlapping intervals built from `TopImpactScorer` scores.
- **Similarity lens:** `SupernodeLens` (sentence-transformer embeddings, PCA-reduced to 1D) clusters nodes *within* each interval.
- **Clustering:** Complete-link thresholding (`cluster_1d_dict`) groups nodes whose semantic values lie within a fixed distance.

#### `llm` — `LLMGroupingPipeline`
Groups nodes by asking an LLM to form semantically coherent clusters in a single call:
- Sends all node labels (indexed) to the LLM with a structured prompt.
- LLM returns group assignments and a concise label for each cluster.
- Accepts an optional `--n_groups_hint` to softly guide the number of groups.
- Implemented in [graph_skel/mapper/llm_grouping.py](graph_skel/mapper/llm_grouping.py).

Both pipelines produce a skeleton `nx.Graph` with the same structure (nodes carry `nodes=[...]` membership and `label`), so all downstream code is interchangeable.

### 3) LLM-Assisted Labeling

LLMs are used in **two separate roles**, each independently switchable between closed-source (OpenAI) and open-source (local HuggingFace) models:

| Role | Triggered by | Controlled by |
|---|---|---|
| **Auto-interpretation** (`rerun_auto_interpretation`) | Empty/missing node labels in `SupernodeLens` or `LLMGroupingPipeline` | `--open_source_labeling` |
| **Umbrella term** (`get_umbrella_term`) | Multi-node supernodes in `save_graph_with_qparams` that have no pre-set label | `--open_source_labeling` |
| **LLM grouping** (`llm_group_nodes`) | `--grouping llm` | `--open_source_grouping` |

> **⚠️ API key required for closed-source models.**
> By default all three LLM roles route requests through the OpenAI API.
> You must set `OPENAI_API_KEY` at the top of [graph_skel/data/supernode_label.py](graph_skel/data/supernode_label.py) before running.
> If you don't have an OpenAI key, pass `--open_source_labeling` and/or `--open_source_grouping` to use a local open-weight model (e.g. DeepSeek, Qwen, Llama) instead — no API key needed, but a GPU with sufficient VRAM is required.

Model configuration (model names, API key) lives in [graph_skel/data/supernode_label.py](graph_skel/data/supernode_label.py) (`CLOSED_SOURCE_MODEL`, `OPEN_SOURCE_MODEL`, `OPENAI_API_KEY`).

### SupernodeLens
Implemented in [graph_skel/lenses/supernode.py](graph_skel/lenses/supernode.py). Encodes each node label with a sentence-transformer model, then reduces the embeddings to a single scalar via PCA. This 1D value is used as the clustering coordinate inside `MapperPipeline`. Accepts `use_closed_source_labeling` to control which LLM is used for re-labeling nodes whose label is empty.

### TopImpactScorer
Implemented in [graph_skel/mapper/top_impact.py](graph_skel/mapper/top_impact.py). Not a lens — a standalone node-selection function that scores every node in the graph by its causal importance for the predicted output token. Used by both pipelines to decide which nodes survive pruning.

  **How it works (high level):**
  1. **Identify sinks.** The top-predicted logit node is the primary target; high-probability alternative logits are also included as contrastive sinks.
  2. **Signed path sums (reverse-topological DP).** For each sink, the algorithm propagates backwards through the graph, tracking how much *positive* (supporting) and *negative* (suppressing) path mass reaches each node. Edge contributions are normalized by the total incoming weight at each node so that relative share of influence is measured rather than raw magnitude.
  3. **Sub-scores per node.** Several interpretability-motivated scores are combined:
     - **Importance** — total positive + negative path mass toward all sinks (weighted by sink probabilities).
     - **Competition** — how strongly a node supports *alternative* (non-top) logits; high competition nodes may be misleading.
     - **Suppress-alt** — how strongly a node *suppresses* alternatives, i.e. actively steers toward the top prediction.
     - **Gateway** — bottleneck nodes whose removal would cut many paths to the sink.
     - **Dominator** — nodes that lie on *every* path from input embeddings to the sink (topological dominance).
     - **Group boost** — small bonus when semantically similar nodes cluster together, rewarding coherent circuits.
  4. **Node-type priors.** A prior multiplier modulates the final score by node type: logit/feature nodes (`1.0`), embedding nodes (`0.5`), error nodes (`0.05`), etc.
  5. **Final score.** `score = prior × (w_importance · importance + w_competition · competition + …)` using fixed linear weights defined in `VNSWeights`.
  6. **Selection threshold.** Nodes whose importance score exceeds an absolute minimum (`min_importance_for_selection = 0.035`) are kept; the rest are pruned.

## Project Structure
- [graph_skel/data/](graph_skel/data/): graph IO, format helpers, LLM labeling utilities, and save utilities
- [graph_skel/lenses/](graph_skel/lenses/): `SupernodeLens` (semantic similarity for clustering)
- [graph_skel/mapper/](graph_skel/mapper/): pruning, embedding-based Mapper pipeline, and LLM-based grouping pipeline
- [graph_skel/experiments/](graph_skel/experiments/): CLI runner
- [graph_skel/external/Faithfulness/](graph_skel/external/Faithfulness/): circuit-tracer dependency

## Setup
Install Python dependencies:
```
pip install -r graph_skel/requirements.txt
```

If you plan to **build graphs from prompts** (instead of using existing JSONs), you also need the `circuit_tracer` package from the bundled submodule:

```
pip install -e graph_skel/external/Faithfulness
```

## Usage

### Run on a single graph (default: embedding-based grouping)
```
python -m graph_skel.experiments.run_experiment --graph graph_skel/data/sample_graphs/<SAMPLE>.json
```

### Run on a directory of graphs
```
python -m graph_skel.experiments.run_experiment --graph_dir graph_skel/data/sample_graphs/
```

### Use LLM-based grouping (closed-source OpenAI by default)
```
python -m graph_skel.experiments.run_experiment \
    --graph graph_skel/data/sample_graphs/<SAMPLE>.json \
    --grouping llm
```

### Use LLM-based grouping with a local open-source model
```
python -m graph_skel.experiments.run_experiment \
    --graph graph_skel/data/sample_graphs/<SAMPLE>.json \
    --grouping llm \
    --open_source_grouping
```

### Use a local open-source model for auto-interpretation and umbrella-term labeling
```
python -m graph_skel.experiments.run_experiment \
    --graph graph_skel/data/sample_graphs/<SAMPLE>.json \
    --open_source_labeling
```

### Mix: open-source labeling + closed-source grouping
```
python -m graph_skel.experiments.run_experiment \
    --graph graph_skel/data/sample_graphs/<SAMPLE>.json \
    --grouping llm \
    --open_source_labeling
```

### Build a graph from a prompt
```
python -m graph_skel.experiments.run_experiment --prompt "Your prompt here"
```

### All CLI flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--graph` | str | — | Path to a single input graph JSON |
| `--graph_dir` | str | — | Path to a directory of graph JSON files |
| `--prompt` | str | — | Build attribution graph from this prompt (requires `circuit_tracer`) |
| `--grouping` | `embedding`\|`llm` | `embedding` | Grouping method: PCA-based Mapper or LLM-based clustering |
| `--open_source_grouping` | flag | off | Use local HuggingFace model for LLM-based **grouping** instead of OpenAI |
| `--open_source_labeling` | flag | off | Use local HuggingFace model for **auto-interpretation** and **umbrella terms** instead of OpenAI |
| `--n_groups_hint` | int | 0 | Soft hint for number of groups when `--grouping llm` (0 = no hint) |
| `--desired_logit_prob` | float | 0.95 | Target cumulative logit probability (prompt mode) |
| `--max_n_logits` | int | 10 | Max logit nodes (prompt mode) |
| `--max_feature_nodes` | int | 8192 | Max feature nodes (prompt mode) |
| `--graph_batch_size` | int | 256 | Batch size for attribution graph construction (prompt mode) |
| `--node_threshold` | float | 0.5 | Node pruning threshold (prompt mode) |
| `--edge_threshold` | float | 0.8 | Edge pruning threshold (prompt mode) |

Exactly one of `--graph`, `--graph_dir`, or `--prompt` must be provided.

## Output

The output is a JSON graph with a `qParams` section that contains:
- `pinnedIds`: all nodes kept in the skeleton
- `supernodes`: grouped clusters with generated labels

The original nodes and edges are preserved, so visualization tools can overlay the supernode structure.

## Data Format (expected input)

Each graph JSON includes:
- `nodes`: list of node objects with fields such as `node_id`, `label`/`clerp`, `feature_type`, `layer`, `ctx_idx`, `token_prob`, etc.
- `links`: list of directed edges with `source`, `target`, `weight`.
- `metadata`: prompt and run metadata (optional but used by the pipeline).

## References
<a id="1">[1]</a>
Rosen, P., Hajij, M., & Wang, B. (2023, October). Homology-preserving multi-scale graph skeletonization using mapper on graphs. In 2023 Topological Data Analysis and Visualization (TopoInVis) (pp. 10-20). IEEE.
