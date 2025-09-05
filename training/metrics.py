import sys
from pathlib import Path

# Add submodule path to Python import path
submodule_path = Path(__file__).resolve().parent.parent / "external" / "Faithfulness"
sys.path.append(str(submodule_path))


from tqdm import tqdm
import torch
from datetime import datetime


def faithfulness_score(model, skeleton_graph):
    prompt = "Fact: the capital of the state containing Dallas is"
    answer = "Texas"
    print("Metrics | Get faithfulness score | ", datetime.now())
    single_graph_faithfulness = model.graph_faithfulness(
        input=prompt,
        completion=answer,
        selected_features=skeleton_graph['feature_nodes'],
        selected_errors=skeleton_graph['error_nodes'],
        mean_ablate=False,
        mean_ablation_samples=1000,
        retain_bos_features=True,
        direct_effects=True,
        freeze_attention=True
    )
    print(f"Fidelity score: {single_graph_faithfulness} | {datetime.now()}")
    return single_graph_faithfulness


def faithfulness_score_dummy(full_graph, skeleton_graph):
    """Calculate faithfulness as node coverage percentage."""
    skeleton_nodes = set()
    for _, data in skeleton_graph.nodes(data=True):
        skeleton_nodes.update(data['nodes'])
    
    return len(skeleton_nodes) / full_graph.number_of_nodes()


def minimality_score(skeleton_graph):
    """Calculate minimality as inverse of skeleton size."""
    n_nodes = skeleton_graph.number_of_nodes()
    return 1.0 / n_nodes if n_nodes > 0 else 0.0