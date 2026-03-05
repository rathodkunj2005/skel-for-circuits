# Graph Skeletonization for Attribution Graphs

This project simplifies attribution graphs in two phases:
1) **Prune** low-importance nodes.
2) **Group** the remaining nodes into **supernodes** using a Mapper-style cover and clustering.

The resulting skeleton is saved back into the original graph format via `qParams` (pinned nodes + supernodes), so downstream visualization tools can render simplified graphs without losing original node metadata.

## How It Works

### 1) Pruning (node reduction)
Pruning is implemented in [graph_skel/mapper/utils.py](graph_skel/mapper/utils.py) and is invoked by [graph_skel/mapper/pipeline.py](graph_skel/mapper/pipeline.py).

Current behavior (default pipeline):
- Removes nodes with non-positive scores from the chosen lens.
- Removes additional low-score nodes using a **tail-aware median** cutoff.
- Uses **TopImpactLens** as the pruning signal (`lenses[2]` in the pipeline).
- Drops MLP reconstruction error nodes during pruning.

Notes:
- `prune_graph_by_one_lens` currently **overrides** `prune_fraction`, `pin_node_to_input_ratio`, and `pinned_count` to `0` internally (see [graph_skel/mapper/utils.py](graph_skel/mapper/utils.py#L380)). This means only the dynamic cutoff is active unless you modify that function.
- Extra logit-node pruning exists but is currently disabled in the default path.

### 2) Supernode construction (grouping)
After pruning, nodes are grouped into supernodes using a Mapper-style cover over a 1D lens:

- **Cover:** The pipeline builds overlapping intervals from lens values (`create_cover`).
- **Primary lens (1D):** `TopImpactLens` is used as the cover coordinate.
- **Similarity lens:** `SupernodeLens` provides semantic similarity values used for clustering *within* each interval.
- **Clustering:** A complete-link style thresholding (`cluster_1d_dict`) groups nodes whose semantic values are within a fixed distance.

The pipeline then creates a skeleton graph where each supernode is a cluster and edges are weighted by the sum of cross-cluster edge weights in the original graph.

### Lens Functions
Implemented under [graph_skel/lenses/](graph_skel/lenses/):
- `DepthLens`: shortest path distance to output.
- `SupernodeLens`: sentence-transformer embeddings of node labels, reduced to 1D (PCA).
- `TopImpactLens`: backward influence propagation with beam pruning.
- `ImportanceLens`: simpler influence score based on node attributes.

## Project Structure
- [graph_skel/data/](graph_skel/data/): graph IO, format helpers, and save utilities
- [graph_skel/lenses/](graph_skel/lenses/): lens functions for pruning and clustering
- [graph_skel/mapper/](graph_skel/mapper/): pruning + Mapper pipeline
- [graph_skel/training/](graph_skel/training/): weight search and metrics
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

### Run on a single graph
```
python -m graph_skel.experiments.run_experiment --graph graph_skel/data/sample_graphs/<SAMPLE>.json
```

### Run on a directory of graphs
```
python -m graph_skel.experiments.run_experiment --graph_dir graph_skel/data/sample_graphs/
```

### (Optional) Train lens weights
```
python -m graph_skel.experiments.run_experiment --do_pretrain True --train_data_dir graph_skel/data/training/ --graph graph_skel/data/sample_graphs/<SAMPLE>.json
```

The runner expects a `training/training_results.json` file. One is included in this repo; training overwrites it.

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
