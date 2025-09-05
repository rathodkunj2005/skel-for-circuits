import numpy as np
from lenses.importance import ImportanceLens

def prune_graph_by_one_lens(G, lens, prune_fraction=0.1):
    scores = {node: lens(G, node) for node in G.nodes()}
    
    # Determine threshold
    n_remove = int(len(scores) * prune_fraction)
    if n_remove == 0:
        return G.copy()  # nothing to remove
    
    # Sort nodes by score (ascending)
    nodes_sorted = sorted(scores.items(), key=lambda x: x[1])
    nodes_to_remove = [node for node, _ in nodes_sorted[:n_remove]]
    
    # Remove nodes
    G_pruned = G.copy()
    G_pruned.remove_nodes_from(nodes_to_remove)
    
    return G_pruned



def prune_graph_by_lens_combination(G, lenses, weights, prune_fraction=0.9):

    # scores = {node: lens(G, node) for node in G.nodes()}
    scores = {}
    # for node in G.nodes():
    #     combined = 0.0
    #     for lens, weight in zip(lenses, weights):
    #         combined += weight * lens(G, node)
    #     scores[node] = combined
    
    # lenses are depth, supernode, importance
    for node in G.nodes():
        scores[node] = 1/(1/lenses[-1](G, node) * lenses[1](G, node))

    
    # Determine threshold
    n_remove = int(len(scores) * prune_fraction)
    if n_remove == 0:
        return G.copy()  # nothing to remove
    
    # Sort nodes by score (ascending)
    nodes_sorted = sorted(scores.items(), key=lambda x: x[1])
    nodes_to_remove = [node for node, _ in nodes_sorted[:n_remove]]
    
    # Remove nodes
    G_pruned = G.copy()
    G_pruned.remove_nodes_from(nodes_to_remove)
    
    return G_pruned

    
def create_cover(points, n_intervals=10, overlap=0.2):
    min_val, max_val = np.min(points), np.max(points)
    range_val = max_val - min_val
    interval_length = range_val / (n_intervals - overlap * (n_intervals - 1))
    step = interval_length * (1 - overlap)
    
    intervals = []
    start = min_val
    for _ in range(n_intervals):
        end = start + interval_length
        intervals.append((start, end))
        start = start + step
    
    return intervals
