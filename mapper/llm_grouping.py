"""
mapper/llm_grouping.py
----------------------
LLM-based alternative to the embedding + 1-D PCA grouping in MapperPipeline.

Instead of projecting label embeddings onto a 1-D axis and bucketing by
distance, this pipeline sends ALL node labels to an LLM in a single call and
asks it to:
  1. Form semantically coherent groups.
  2. Assign a concise representative label to each group.

The returned skeleton graph is structurally identical to the one produced by
MapperPipeline, so all downstream code (loaders.py, graph rendering, etc.)
works without modification.

Reuses from data/supernode_label.py:
  - rerun_auto_interpretation()  – fill missing labels (same as SupernodeLens)
  - generate_text()              – unified LLM call helper
  - parse_json_response()        – robust JSON extractor (used for single-key objects)
"""

import json
import re
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .utils import prune_graph_by_one_lens
from data.loaders import get_top_logit_node
from data.supernode_label import (
    generate_text,
    rerun_auto_interpretation,
)


# ---------------------------------------------------------------------------
# Label helpers (mirrors the cleaning logic in SupernodeLens._compute_lens_values)
# ---------------------------------------------------------------------------

def _clean_label(raw: str) -> str:
    """
    Apply the same cleaning steps used by SupernodeLens:
      1. Strip "Emb: " prefix.
      2. Remove a leading [bracketed] annotation (expert / auto-interp tags).
    """
    cleaned = raw.replace("Emb: ", "").strip()
    cleaned = re.sub(r'^\[[^\]]*\]\s*', '', cleaned)
    return cleaned


def _resolve_labels(G: nx.Graph, use_closed_source_labeling: bool = True) -> Dict[str, str]:
    """
    Return {node_id: label} for every node in G.

    - Applies _clean_label to existing labels.
    - For nodes whose label is empty after cleaning, calls
      rerun_auto_interpretation() (same fallback as SupernodeLens).
    - Updates G's node data in-place so downstream consumers see the resolved
      labels (mirrors the behaviour of SupernodeLens._compute_lens_values).

    Args:
        use_closed_source_labeling: Passed to rerun_auto_interpretation — True
            uses the OpenAI API; False uses the local open-source model.
    """
    labels: Dict[str, str] = {}
    for node, data in G.nodes(data=True):
        raw = str(data.get('label', '') or '')
        cleaned = _clean_label(raw)
        if cleaned:
            labels[node] = cleaned
            data['label'] = cleaned
        else:
            feature_id = data.get('feature', None)
            label = rerun_auto_interpretation(feature_id, use_closed_source=use_closed_source_labeling) if feature_id else "[unlabeled]"
            labels[node] = label
            data['label'] = label
    return labels


# ---------------------------------------------------------------------------
# LLM grouping
# ---------------------------------------------------------------------------

def _parse_groups_response(raw: str) -> Optional[List[dict]]:
    """
    Extract the list of group dicts from the LLM's JSON response.

    Expected shape:
        {"groups": [{"label": "...", "indices": [0, 2, 5]}, ...]}

    Returns the list on success, None on parse failure.
    Strips <think>…</think> blocks emitted by open-source reasoning models.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try fenced code block first
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    candidate = fence.group(1) if fence else None

    # Fallback: last bare {...} block
    if candidate is None:
        matches = list(re.finditer(r"\{.*\}", cleaned, re.DOTALL))
        if matches:
            candidate = matches[-1].group(0)

    if candidate:
        try:
            obj = json.loads(candidate)
            groups = obj.get("groups", None)
            if isinstance(groups, list) and all(
                isinstance(g, dict) and "indices" in g for g in groups
            ):
                return groups
        except json.JSONDecodeError:
            pass

    return None


def llm_group_nodes(
    node_labels: Dict[str, str],
    use_closed_source: bool = True,
    n_groups_hint: Optional[int] = None,
) -> List[Tuple[str, List[str]]]:
    """
    Ask an LLM to semantically cluster nodes and name each cluster.

    The LLM receives a stable integer index → label mapping so node IDs
    (which may be arbitrary strings) never appear in the prompt.

    Args:
        node_labels:        {node_id: label_string} for every node to group.
        use_closed_source:  True → OpenAI API; False → local HF model.
        n_groups_hint:      Optional soft hint for the number of groups.

    Returns:
        List of (group_label, [node_ids]) in LLM-chosen order.
        Any nodes omitted by the LLM are appended as singleton groups.
    """
    if not node_labels:
        return []

    # Stable mapping: int index ↔ node_id
    index_to_node: Dict[int, str] = {i: n for i, n in enumerate(node_labels)}
    node_to_index: Dict[str, int] = {n: i for i, n in index_to_node.items()}

    lines = [f"  [{i}] {node_labels[node]}" for i, node in index_to_node.items()]
    numbered_list = "\n".join(lines)

    hint_str = f" Aim for roughly {n_groups_hint} groups." if n_groups_hint else ""

    prompt = (
        "You are analysing a computational attribution graph for a language model. "
        "Each numbered item below is a node in the graph described by its feature label.\n\n"
        "Your task: group these nodes into semantically coherent clusters and assign each "
        "cluster a concise representative label that captures the shared functional meaning "
        "of its members.\n\n"
        "Rules:\n"
        "- Every node index must appear in exactly one group.\n"
        "- Nodes that fire on unrelated concepts should be in separate groups.\n"
        "- Nodes that are functionally identical or very similar should be merged.\n"
        f"- Prefer between 2 and 8 groups.{hint_str}\n"
        "- Group labels must be short phrases (≤6 words), specific, not generic.\n\n"
        f"Nodes:\n{numbered_list}\n\n"
        "Respond ONLY with a single valid JSON object — no prose, no markdown fences.\n"
        "The JSON must have exactly one field:\n"
        '  "groups": a list of objects, each with:\n'
        '    "label": the concise group label (string),\n'
        '    "indices": the list of node indices in this group (list of ints).\n\n'
        'Example:\n'
        '{"groups": ['
        '{"label": "clause-final punctuation", "indices": [0, 3, 7]}, '
        '{"label": "monetary values", "indices": [1, 2, 4, 5, 6]}'
        ']}\n'
    )

    def _call(p: str):
        raw = generate_text(p, max_tokens=4096, temperature=0.7,
                            use_closed_source=use_closed_source)
        return _parse_groups_response(raw)

    groups_raw = _call(prompt)

    if groups_raw is None:
        print("[LLMGrouping] JSON parse failed on first attempt — retrying with strict prompt.")
        groups_raw = _call(
            prompt
            + "\nIMPORTANT: Your previous response could not be parsed. "
            "You MUST respond with ONLY a raw JSON object matching the schema above.\n"
        )

    if groups_raw is None:
        print("[LLMGrouping] Warning: both attempts failed — putting all nodes in one group.")
        return [("[unlabeled]", list(node_labels.keys()))]

    # Map integer indices → node IDs, guard against duplication / out-of-range
    result: List[Tuple[str, List[str]]] = []
    seen: set = set()
    for g in groups_raw:
        label = str(g.get("label", "[unlabeled]")).strip() or "[unlabeled]"
        raw_indices = g.get("indices", [])
        node_ids = []
        for idx in raw_indices:
            idx = int(idx)
            if idx in index_to_node and idx not in seen:
                seen.add(idx)
                node_ids.append(index_to_node[idx])
        if node_ids:
            result.append((label, node_ids))

    # Append any nodes the LLM forgot as singletons
    for i, node in index_to_node.items():
        if i not in seen:
            result.append((node_labels[node], [node]))

    return result


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class LLMGroupingPipeline:
    """
    Drop-in replacement for MapperPipeline that uses LLM judgement instead of
    embedding-based 1-D PCA to form supernodes.

    The returned skeleton nx.Graph is structurally identical to
    MapperPipeline's output:
      - Each skeleton node i carries  nodes=[original_node_id, ...]
                                      label="LLM-assigned group label"
      - Skeleton edges carry the aggregated weight of cross-cluster edges.

    Because the skeleton node already has a meaningful `label`, the downstream
    loaders.py helper honours it and skips the redundant get_umbrella_term call
    (see the patched `prepare_and_send_json` function in data/loaders.py).

    Args:
        lenses:             [SupernodeLens-compatible, TopImpactLens] — only
                            lenses[1] (the pruning lens) is actually used here.
        use_closed_source:  True → OpenAI API; False → local HF model.
        n_groups_hint:      Optional soft hint for number of groups to request.
    """

    def __init__(
        self,
        lenses,
        use_closed_source: bool = True,
        use_closed_source_labeling: bool = True,
        n_groups_hint: Optional[int] = None,
    ) -> None:
        self.lenses = lenses
        self.use_closed_source = use_closed_source
        self.use_closed_source_labeling = use_closed_source_labeling
        self.n_groups_hint = n_groups_hint

    def _prune(self, G: nx.Graph) -> nx.Graph:
        print(f"LLMGroupingPipeline | pruning starts with {G}")
        return prune_graph_by_one_lens(G, self.lenses[1])

    def __call__(self, G: nx.Graph) -> nx.Graph:
        original_G = G
        G = self._prune(G)

        # 1. Resolve labels: clean + rerun auto-interp for empty labels
        node_labels = _resolve_labels(G, use_closed_source_labeling=self.use_closed_source_labeling)

        # Propagate any label changes back to the original (pre-prune) graph
        for node, data in G.nodes(data=True):
            if node in original_G.nodes and 'label' in data:
                original_G.nodes[node]['label'] = data['label']

        # 2. Keep the top-output node as its own supernode (mirrors MapperPipeline)
        top_output_node = get_top_logit_node(G)
        remaining_labels = {n: l for n, l in node_labels.items()
                            if n != top_output_node}

        # 3. LLM-based grouping on all remaining nodes
        if remaining_labels:
            groups = llm_group_nodes(
                remaining_labels,
                use_closed_source=self.use_closed_source,
                n_groups_hint=self.n_groups_hint,
            )
        else:
            groups = []

        # Top-output node is cluster 0 (same convention as MapperPipeline)
        top_label = node_labels.get(top_output_node, "[output]")
        all_groups: List[Tuple[str, List[str]]] = (
            [(top_label, [top_output_node])] + groups
        )

        print("LLMGroupingPipeline | Groups:",
              [(lbl, len(nodes)) for lbl, nodes in all_groups])

        # 4. Build skeleton graph (same structure as MapperPipeline)
        skeleton = nx.Graph()
        for i, (group_label, cluster) in enumerate(all_groups):
            skeleton.add_node(i, nodes=cluster, label=group_label)

        for i in range(len(all_groups)):
            for j in range(i + 1, len(all_groups)):
                weight = sum(
                    G[u][v].get('weight', 1.0)
                    for u in all_groups[i][1]
                    for v in all_groups[j][1]
                    if G.has_edge(u, v)
                )
                if weight > 0:
                    skeleton.add_edge(i, j, weight=weight)

        print("LLMGroupingPipeline | skeleton:", skeleton)
        return skeleton
