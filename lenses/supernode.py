import re
from typing import Dict, List

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from .base import LensFunction
from data.loaders import get_input_nodes
from data.supernode_label import rerun_auto_interpretation

class SupernodeLens(LensFunction):
    def __init__(
        self,
        model_name: str = "sentence-transformers/gtr-t5-large",
        use_closed_source_labeling: bool = True,
    ) -> None:
        """
        Initializes the SemanticLens.

        Args:
            model_name (str): The name of the SentenceTransformer model to use for generating label embeddings.
            nvidia/NV-Embed-v1
            Alibaba-NLP/gte-large-en-v1.5
            all-mpnet-base-v2
            all-mpnet-base-v3
            BAAI/bge-large-en-v1.5
            sentence-transformers/gtr-t5-large
            use_closed_source_labeling (bool): If True, use the closed-source (OpenAI) model for
                rerun_auto_interpretation; otherwise use the local open-source model.
        """
        self.model = SentenceTransformer(model_name)
        self.pca = PCA(n_components=1)
        self.use_closed_source_labeling = use_closed_source_labeling
        self._lens_values: Dict[str, float] | None = None
        self._graph_hash: str | None = None  # Used for simple caching to avoid re-computation

    def _clean_and_unique(self, text: str) -> str:
        # 1. Remove brackets
        clean = re.sub(r'[\[\]]', '', text)
        
        # 2. Split into words
        words = clean.split()
        
        # 3. Deduplicate (preserve order)
        seen = set()
        unique = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        
        # 4. Join back
        return " ".join(unique)
    
    def _ensure_lens_values(self, G: nx.Graph) -> None:
        """Recompute lens values if we haven't seen this graph instance."""
        graph_hash = nx.weisfeiler_lehman_graph_hash(G)
        if self._graph_hash != graph_hash:
            self._compute_lens_values(G)
            self._graph_hash = graph_hash

    def _compute_lens_values(self, G: nx.Graph) -> None:
        """
        Computes the lens values for all nodes in the graph based on the semantic
        embeddings of their labels.
        """
        if G.number_of_nodes() == 0:
            self._lens_values = {}
            return

        # Get the list of labels for all nodes in the graph
        labels: List[str] = []
        print("--> SUPERNODE LENS: Graph id:", id(G), "Number of nodes:", G.number_of_nodes())
        scan = G.graph.get("scan", "")
        np_source_set = G.graph.get("neuronpedia_source_set", "")
        for _, data in G.nodes(data=True):
            raw = str(data.get('label', '') or '')
            cleaned1 = raw.replace("Emb: ", "").strip()
            # remove leading [bracketed] content because it often comes from expert annotations
            cleaned = re.sub(r'^\[[^\]]*\]\s*', '', cleaned1)
            if cleaned != "":
                labels.append(cleaned)
                data['label'] = cleaned  # update the graph with the cleaned label
            else:
                feature_id = data.get('feature', None)
                label = rerun_auto_interpretation(
                    feature_id,
                    use_closed_source=self.use_closed_source_labeling,
                    scan=scan,
                    layer=str(data.get('layer', '')),
                    feature_type=data.get('feature_type', ''),
                    js_node_id=data.get('jsNodeId', ''),
                    np_source_set=np_source_set,
                ) if feature_id else "[unlabeled]"
                labels.append(label)
                data['label'] = label  # update the graph with the new label

        # Generate embeddings for all labels
        embeddings = self.model.encode(labels, normalize_embeddings=True)

        ######################################### PCA #########################################
        # Fit PCA on the embeddings and transform them to a 1D representation
        lens_1d = self.pca.fit_transform(embeddings).ravel()
        #######################################################################################

        # Create a dictionary mapping each node to its computed lens value
        self._lens_values = {node: lens_1d[i] for i, node in enumerate(G.nodes())}

        # Temp Test: Amplify the lens values of input nodes because they are often independtly analyzable
        input_nodes = get_input_nodes(G)
        for node in input_nodes:
            if node in self._lens_values:
                self._lens_values[node] *= 5
            

    def __call__(self, G: nx.Graph, node):
        """
        Calculates and returns the lens value for a specific node.
        For efficiency, it computes the lens for all nodes on the first call for a given graph
        and caches the result.
        """
        self._ensure_lens_values(G)
        if self._lens_values is None:
            return 0.0
        return 2 + self._lens_values.get(node, 0.0)

    def get_all_lens_values(self, G: nx.Graph) -> Dict[str, float]:
        """
        A helper function to compute and retrieve the lens values for all nodes in the graph.
        This is often a more practical approach for use with Mapper algorithms.
        """
        self._ensure_lens_values(G)
        return dict(self._lens_values or {})
