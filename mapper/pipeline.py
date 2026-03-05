import networkx as nx
import numpy as np
from .utils import create_cover, prune_graph_by_one_lens
from data.loaders import get_top_logit_node

class MapperPipeline:
    def __init__(
            self,
            lenses,
            n_intervals=5,
            overlap=0.001,
            similarity_cluster_threshold=0.3,
        ):
        self.lenses = lenses
        # lenses[0]: SupernodeLens  (semantic similarity)
        # lenses[1]: TopImpactLens  (node scorer / pruner)

        self.n_intervals = n_intervals
        self.overlap = overlap
        self.semantic_values = {}
        self.similarity_cluster_threshold = similarity_cluster_threshold
    
    def _prune(self, G):
        print(f"Pipeline | pruning starts with {G}")
        G = prune_graph_by_one_lens(G, self.lenses[1])
        return G

    def _compute_lens(self, G):
        lens_values = {}
        print("Pipeline | pruned graph:", G)
        for node in G.nodes():
            lens_values[node] = self.lenses[1](G, node)  # TopImpactLens score
            self.semantic_values[node] = self.lenses[0](G, node)  # SupernodeLens value
        return lens_values
    
    def cluster_1d_dict(self, items, score_key="score", threshold=0.2):
        """
        items: list of dicts, each containing {"node": ..., score_key: <float>}
        threshold: maximum allowed distance between ANY two items in one cluster
        """
        
        # Sort by the 1D score
        items_sorted = sorted(items, key=lambda x: x[score_key])

        clusters = []
        current_cluster = [items_sorted[0]["node"]]

        # Track min and max score within current cluster
        cluster_min = items_sorted[0][score_key]
        cluster_max = items_sorted[0][score_key]

        for item in items_sorted[1:]:

            s = item[score_key]

            # Check complete-link condition:
            # The cluster remains valid only if max - min <= threshold
            new_min = min(cluster_min, s)
            new_max = max(cluster_max, s)

            if (new_max - new_min) <= threshold:
                # Still tight — add to current cluster
                current_cluster.append(item["node"])
                cluster_min, cluster_max = new_min, new_max
            else:
                # Too far — finalize current cluster, start new one
                clusters.append(current_cluster)
                current_cluster = [item["node"]]
                cluster_min = cluster_max = s

        # Add final cluster
        clusters.append(current_cluster)

        return clusters



    def __call1D__(self, G):
        original_G = G  # keep reference before _prune makes a copy
        G = self._prune(G)
    
        lens_values = self._compute_lens(G)

        # Propagate label changes made by SupernodeLens back to the original graph
        # (_prune returns a copy, so updates to G are not visible in original_G)
        for node, data in G.nodes(data=True):
            if node in original_G.nodes and 'label' in data:
                original_G.nodes[node]['label'] = data['label']

        points = np.array(list(lens_values.values())).reshape(-1, 1)
        top_output_node = get_top_logit_node(G)
        intervals = create_cover(points, self.n_intervals, self.overlap)
        
        # Keep logit node (top_output_node) as a separate cluster
        clusters = [[top_output_node]]
        
        print("----"*10)
        print("similarity_cluster_threshold = ", self.similarity_cluster_threshold)
        print("----"*10)
        
        for interval in intervals:
            nodes_in_interval = [
                node for node, value in lens_values.items() 
                if interval[0] <= value < interval[1]
            ]
            if not nodes_in_interval:
                continue
            
            if top_output_node in nodes_in_interval:
                nodes_in_interval.remove(top_output_node)

            # Remove nodes already assigned to a cluster
            for n in nodes_in_interval:
                del lens_values[n]

            if len(nodes_in_interval) == 1:
                clusters.append(nodes_in_interval)
            else:
                d = [{"node": node, "score": self.semantic_values[node]} for node in nodes_in_interval]
                cluster = self.cluster_1d_dict(d, threshold=self.similarity_cluster_threshold)
                for c in cluster:
                    if c not in clusters:
                        clusters.append(c)
        
        print("Pipeline | Supernodes:", clusters)
        
        # Create skeleton graph
        skeleton = nx.Graph()
        for i, cluster in enumerate(clusters):
            skeleton.add_node(i, nodes=cluster)

        # Add edges between clusters
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                weight = sum(G[u][v].get('weight', 1.0) for u in clusters[i] for v in clusters[j] if G.has_edge(u, v))
                if weight > 0:
                    skeleton.add_edge(i, j, weight=weight)

        print("Pipeline | skeleton:", skeleton)
        return skeleton
    
    
    def __call__(self, G):
        return self.__call1D__(G)

