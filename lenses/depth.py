import networkx as nx
from .base import LensFunction
from data.loaders import get_output_nodes, get_input_nodes

class DepthLens(LensFunction):
    def __init__(self, reverse=True):
        self.reverse = reverse  # True for distance to output
        
    def __call__(self, G, node):
        output_nodes = get_output_nodes(G)
        input_nodes = get_input_nodes(G)
        if node in output_nodes or node in input_nodes:
            return 0.0
        
        # Reverse graph for distance to output
        if self.reverse:
            G = G.reverse()
        
        # Compute shortest path distances
        min_dist = float('inf')
        for output in output_nodes:
            try:
                dist = nx.shortest_path_length(G, source=node, target=output, weight=None)
                if dist < min_dist:
                    min_dist = dist
            except nx.NetworkXNoPath:
                continue
        
        return min_dist if min_dist != float('inf') else -1.0
