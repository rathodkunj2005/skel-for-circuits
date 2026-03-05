
from data.loaders import get_top_logit_node, get_input_nodes
import networkx as nx
from collections import Counter, defaultdict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


import re
from typing import Any


from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple
import math



@dataclass(frozen=True)
class VNSWeights:
    """Linear weights for combining sub-scores into final score."""
    importance: float = 1.0
    competition: float = 0.7
    suppress_alt: float = 0.5
    gateway: float = 0.2
    dominator: float = 0.2
    suppress_top: float = 0.1
    group_boost: float = 0.05


@dataclass(frozen=True)
class VNSPriors:
    """
    interpretability priors.
    """
    # Priors by (normalized) node_type
    type_prior: Mapping[str, float] = field(
        default_factory=lambda: {
            "logit": 1.0,
            "feature": 1.0,
            "embedding": 0.5,
            "unexplored": 0.7,
            "error": 0.05,
            "other": 0.6,
        }
    )

    # If label is empty/unknown
    empty_label_prior: float = 0.6

    # If label looks like generic assistant boilerplate
    assistant_label_prior: float = 0.3

    # If embedding token looks like formatting/punctuation
    formatting_embedding_prior: float = 0.6


@dataclass(frozen=True)
class VNSAltPolicy:
    """How to choose alternative logits for contrastive scoring."""
    enabled: bool = True
    max_alts: int = 3
    min_alt_prob: float = 0.05          # include alt if p >= this
    trigger_if_top_prob_below: float = 0.85  # if top prob < this, enable alts more readily
    trigger_if_second_prob_above: float = 0.10


@dataclass(frozen=True)
class VNSConfig:
    # Attribute names
    edge_weight_attr: str = "weight"
    node_type_attr: str = "feature_type"
    node_prob_attr: str = "token_prob"
    node_label_attr: str = "clerp"

    # Numeric stability
    eps: float = 1e-12

    # Group-boost shape: cluster_mass / (cluster_size^alpha)
    group_alpha: float = 0.5

    # Contrastive weight scaling for alts
    lambda_alt: float = 0.5

    # Root selection (for dominance): prefer embeddings; fallback to in_degree==0
    dominance_prefer_embeddings: bool = True

    min_importance_for_selection: float = 0.035  # Minimum importance score to consider a node for selection
    selection_importance_fraction: float = 0.65   # Fraction of max importance score to use as a threshold for selection
    
    # Heuristic stoplist for assistant boilerplate in labels
    assistant_label_regex: str = r"\bassistant\b|\bsystem\b|\byou are\b|\bhelpful\b"

    # Heuristic for bracket tag extraction like "[harmful requests]" or "[sqrt(x)]"
    bracket_tag_regex: str = r"^\[([^\]]+)\]"

    # Jaccard similarity threshold for group boosting based on label similarity
    group_similarity_threshold: float = 0.85

    # Default HuggingFace sentence embedding model for semantic grouping
    text_embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    
    # If you want to hard-exclude error nodes from selection, keep scoring but filter later
    hard_exclude_error_nodes_in_selection: bool = True

    
@dataclass
class NodeScoreBreakdown:
    """Per-node breakdown of all sub-scores + priors."""
    prior: float
    importance: float
    support: float
    suppress_top: float
    competition: float
    suppress_alt: float
    gateway: float
    dominator: float
    group_boost: float
    score: float


class TopImpactLens:
    """
    Vignette node selector for pruned attribution graphs (DAGs).

    Assumptions:
      - G is a nx.DiGraph (acyclic preferred)
      - edges go from lower layers -> higher layers (toward logit nodes)
      - edge attribute `edge_weight_attr` is signed (support/suppress)
      - node attributes include type/prob/label (configurable)
    """

    def __init__(
        self,
        config: Optional[VNSConfig] = None,
        weights: Optional[VNSWeights] = None,
        priors: Optional[VNSPriors] = None,
        alt_policy: Optional[VNSAltPolicy] = None,
    ) -> None:
        self.cfg = config or VNSConfig()
        self.w = weights or VNSWeights()
        self.priors = priors or VNSPriors()
        self.alt_policy = alt_policy or VNSAltPolicy()

        self.cache = {}
        self._text_embedding_model = None

    def __call__(self, G: nx.DiGraph, node):
        if id(G) not in self.cache:
            self.G = G
            self._topo: List[Any] = list(nx.topological_sort(self.G))
            self._rev_topo: List[Any] = list(reversed(self._topo))

            # Cached per-edge normalized weights/sign
            self._norm_abs: Dict[Tuple[Any, Any], float] = {}
            self._sign: Dict[Tuple[Any, Any], int] = {}

            self._build_incoming_normalization()
            
        
            self.select_nodes()
        
        scores = self.cache[id(G)]
        return scores.get(node, 0.0)
    
    # -----------------------------
    # Public API
    # -----------------------------
    def score_nodes(
        self,
        target_logit: Optional[Any] = None,
        candidate_logits: Optional[Sequence[Any]] = None,
    ) -> Dict[Any, NodeScoreBreakdown]:
        """
        Compute per-node scores + breakdown.

        Args:
          target_logit:
            If provided, score explanations primarily for this sink logit.
            If None, will auto-pick best logit by node_prob_attr.
          candidate_logits:
            Optional explicit list of logit nodes to consider (top-k).
            If None, auto-detect all logit nodes.

        Returns:
          dict: node -> NodeScoreBreakdown
        """
        logits = list(candidate_logits) if candidate_logits is not None else self._find_logit_nodes()
        if not logits:
            raise ValueError("No logit nodes found (or provided).")

        y_star, alts = self._choose_sinks(logits, target_logit=target_logit)
        sinks = [y_star] + alts
        omega = self._sink_weights(y_star, alts)

        # Signed path-sums to each sink
        Dp, Dm = self._signed_path_sums(sinks)  # maps sink -> dict(node->value)

        # Subscores per node
        importance = self._importance(Dp, Dm, omega)
        support = {u: (Dp[y_star].get(u, 0.0) - Dm[y_star].get(u, 0.0)) for u in self.G.nodes()}
        suppress_top = {u: Dm[y_star].get(u, 0.0) for u in self.G.nodes()}

        competition, suppress_alt = self._contrastive_scores(y_star, alts, Dp, Dm, omega)
        gateway = self._gateway_scores(sinks, Dp, Dm, omega)
        dominator = self._dominator_scores(sinks)
        group_boost = self._group_boost_scores(importance)

        # Priors
        prior = {u: self._prior(u) for u in self.G.nodes()}

        # Final score = prior * linear combo
        out: Dict[Any, NodeScoreBreakdown] = {}
        for u in self.G.nodes():
            s = prior[u] * (
                self.w.importance * importance[u]
                + self.w.competition * competition[u]
                + self.w.suppress_alt * suppress_alt[u]
                + self.w.gateway * gateway[u]
                + self.w.dominator * dominator[u]
                + self.w.suppress_top * suppress_top[u]
                + self.w.group_boost * group_boost[u]
            )
            out[u] = NodeScoreBreakdown(
                prior=prior[u],
                importance=importance[u],
                support=support[u],
                suppress_top=suppress_top[u],
                competition=competition[u],
                suppress_alt=suppress_alt[u],
                gateway=gateway[u],
                dominator=dominator[u],
                group_boost=group_boost[u],
                score=s,
            )
        
        print("\n------------\n\n\n\nTopImpactLens: Node Score Breakdown:")

        return out

    def select_nodes(
        self,
        budget: int = 70,
        min_embeddings: int = 1,
        target_logit: Optional[Any] = None,
        candidate_logits: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        """
        Select a node subset under a budget using score ranking + hard constraints:
          - include sinks (target logit + chosen alternatives)
          - include at least min_embeddings embedding nodes
          - then fill remaining budget by descending Score

        Returns:
          List of selected node IDs.
        """
        breakdown = self.score_nodes(target_logit=target_logit, candidate_logits=candidate_logits)

        logits = list(candidate_logits) if candidate_logits is not None else self._find_logit_nodes()
        y_star, alts = self._choose_sinks(logits, target_logit=target_logit)
        sinks = [y_star] + alts

        selected: List[Any] = []
        seen: Set[Any] = set()

        def add(u: Any) -> None:
            if u in self.G and u not in seen:
                selected.append(u)
                seen.add(u)

        # 1) Always include sinks
        for k in sinks:
            add(k)

        # 2) Ensure some embeddings are included
        emb_nodes = [u for u in self.G.nodes() if self._node_type_norm(u) == "embedding"]
        emb_nodes_sorted = sorted(
            emb_nodes,
            key=lambda u: breakdown[u].prior * breakdown[u].importance,
            reverse=True,
        )
        for u in emb_nodes_sorted[: max(0, min_embeddings)]:
            add(u)

        # 3) Fill remaining slots by score, optionally filtering error nodes
        candidates = list(self.G.nodes())
        print(f"Total nodes before filtering: {len(candidates)}")
        if self.cfg.hard_exclude_error_nodes_in_selection:
            candidates = [u for u in candidates if self._node_type_norm(u) != "error"]

        candidates_sorted = sorted(candidates, key=lambda u: breakdown[u].score, reverse=True)
        
        print(f"\n\n---\nSelect_node: Candidate nodes sorted by score: (count={len(candidates_sorted)})")
        

        # ---- absolute importance target ----
        # A. absolute minimum importance threshold for selection
        for u in candidates_sorted:
            if len(selected) >= budget:
                break
            if breakdown[u].importance >= self.cfg.min_importance_for_selection:
                # print(f"adding node {u} with importance {breakdown[u].importance:.4f} and score {breakdown[u].score:.4f}")
                add(u)
        print(f"---------->\t\t\t\t\t\t\t\t\t\t\tSelected {len(selected)} nodes under budget {budget}.")

        
        # ---- cumulative importance target ----  --> sensitive; we use absolute threshold instead for now
        # B. importance coverage: keep adding nodes by score until we reach the target importance coverage
        # target_fraction = self.cfg.selection_importance_fraction  # e.g. 0.8

        # total_importance = sum(
        #     breakdown[u].importance for u in candidates_sorted
        # )

        # target_importance = target_fraction * total_importance
        # cumulative_importance = 0.0

        # for u in candidates_sorted:
        #     if len(selected) >= budget:
        #         break
        #     add(u)
        #     cumulative_importance += breakdown[u].importance
        #     if cumulative_importance >= target_importance:
        #         break
        # print(f"---------->\t\t\t\t\t\t\t\t\t\t\tSelected {len(selected)} nodes under budget {budget}.")

        
        final_scores = {}
        for node in list(breakdown.keys()):
            if node not in selected:
                final_scores[node] = -100
            else:
                final_scores[node] = breakdown[node].score
        
        self.cache[id(self.G)] = final_scores
        return final_scores

    # -----------------------------
    # Normalization + signed path sums
    # -----------------------------
    def _build_incoming_normalization(self) -> None:
        """
        Build incoming-normalized abs weights a_uv = |w_uv| / sum_{p->v} |w_pv|.
        Also stores sign(w_uv).
        """
        w_attr = self.cfg.edge_weight_attr
        eps = self.cfg.eps

        incoming_abs_sum: Dict[Any, float] = {v: 0.0 for v in self.G.nodes()}
        for u, v, data in self.G.edges(data=True):
            w = float(data.get(w_attr, 0.0))
            incoming_abs_sum[v] += abs(w)

        for u, v, data in self.G.edges(data=True):
            w = float(data.get(w_attr, 0.0))
            denom = max(incoming_abs_sum[v], eps)
            self._norm_abs[(u, v)] = abs(w) / denom
            self._sign[(u, v)] = 1 if w >= 0 else -1

    def _signed_path_sums(
        self,
        sinks: Sequence[Any],
    ) -> Tuple[Dict[Any, Dict[Any, float]], Dict[Any, Dict[Any, float]]]:
        """
        For each sink k, compute:
          Dp[k][u] = total positive-parity path mass u->k
          Dm[k][u] = total negative-parity path mass u->k

        Uses reverse topological DP (works because DAG).
        """
        eps = self.cfg.eps

        Dp: Dict[Any, Dict[Any, float]] = {}
        Dm: Dict[Any, Dict[Any, float]] = {}

        for k in sinks:
            dp: Dict[Any, float] = {k: 1.0}
            dm: Dict[Any, float] = {k: 0.0}

            # reverse topo: sinks first, sources last
            for u in self._rev_topo:
                plus_u = dp.get(u, 0.0)
                minus_u = dm.get(u, 0.0)

                # accumulate from successors (u -> v)
                for v in self.G.successors(u):
                    a = self._norm_abs.get((u, v), 0.0)
                    if a <= eps:
                        continue
                    sgn = self._sign.get((u, v), 1)
                    plus_v = dp.get(v, 0.0)
                    minus_v = dm.get(v, 0.0)

                    if sgn > 0:
                        plus_u += a * plus_v
                        minus_u += a * minus_v
                    else:
                        plus_u += a * minus_v
                        minus_u += a * plus_v

                if plus_u != 0.0:
                    dp[u] = plus_u
                if minus_u != 0.0:
                    dm[u] = minus_u

            Dp[k] = dp
            Dm[k] = dm

        return Dp, Dm

    # -----------------------------
    # Subscores
    # -----------------------------
    def _importance(
        self,
        Dp: Dict[Any, Dict[Any, float]],
        Dm: Dict[Any, Dict[Any, float]],
        omega: Mapping[Any, float],
    ) -> Dict[Any, float]:
        """
        Fidelity(u) = sum_k omega_k * (Dp_k(u) + Dm_k(u))  (unsigned involvement)
        """
        out: Dict[Any, float] = {u: 0.0 for u in self.G.nodes()}
        for k, w in omega.items():
            dp = Dp.get(k, {})
            dm = Dm.get(k, {})
            for u in self.G.nodes():
                out[u] += w * (dp.get(u, 0.0) + dm.get(u, 0.0))
        return out

    def _contrastive_scores(
        self,
        y_star: Any,
        alts: Sequence[Any],
        Dp: Dict[Any, Dict[Any, float]],
        Dm: Dict[Any, Dict[Any, float]],
        omega: Mapping[Any, float],
    ) -> Tuple[Dict[Any, float], Dict[Any, float]]:
        """
        If alts exist:
          net_k(u) = Dp_k(u) - Dm_k(u)
          Competition(u) = sum_a omega_a * |net_y*(u) - net_a(u)|
          SuppressAlt(u) = sum_a omega_a * Dm_a(u)
        Else:
          both are zero.
        """
        comp: Dict[Any, float] = {u: 0.0 for u in self.G.nodes()}
        sup_alt: Dict[Any, float] = {u: 0.0 for u in self.G.nodes()}

        if not alts:
            return comp, sup_alt

        dp_y = Dp.get(y_star, {})
        dm_y = Dm.get(y_star, {})

        for a in alts:
            w = omega.get(a, 0.0)
            dp_a = Dp.get(a, {})
            dm_a = Dm.get(a, {})
            for u in self.G.nodes():
                net_y = dp_y.get(u, 0.0) - dm_y.get(u, 0.0)
                net_a = dp_a.get(u, 0.0) - dm_a.get(u, 0.0)
                comp[u] += w * abs(net_y - net_a)
                sup_alt[u] += w * dm_a.get(u, 0.0)

        return comp, sup_alt

    def _gateway_scores(
        self,
        sinks: Sequence[Any],
        Dp: Dict[Any, Dict[Any, float]],
        Dm: Dict[Any, Dict[Any, float]],
        omega: Mapping[Any, float],
    ) -> Dict[Any, float]:
        """
        Gateway entropy score:
          For each u, per sink k:
            f_{u->v}^k = a_uv * D_abs^k(v)
            p_{u->v}^k = f / sum f
            H = -sum p log p
            score_k = 1 - H/log(outdeg)
          Gateway(u) = sum_k omega_k * score_k
        """
        eps = self.cfg.eps
        out: Dict[Any, float] = {u: 0.0 for u in self.G.nodes()}

        for k in sinks:
            w_k = omega.get(k, 0.0)
            if w_k <= 0:
                continue
            dp = Dp.get(k, {})
            dm = Dm.get(k, {})

            for u in self.G.nodes():
                succ = list(self.G.successors(u))
                d = len(succ)
                if d <= 1:
                    # If only one (or zero) outgoing option, it's maximally "focused" if it has any outflow.
                    if d == 1:
                        v = succ[0]
                        flow = self._norm_abs.get((u, v), 0.0) * (dp.get(v, 0.0) + dm.get(v, 0.0))
                        out[u] += w_k * (1.0 if flow > eps else 0.0)
                    continue

                flows: List[float] = []
                for v in succ:
                    a_uv = self._norm_abs.get((u, v), 0.0)
                    dabs_v = dp.get(v, 0.0) + dm.get(v, 0.0)
                    flows.append(a_uv * dabs_v)

                total = sum(flows)
                if total <= eps:
                    continue

                # entropy of normalized flows
                H = 0.0
                for f in flows:
                    p = f / total
                    if p > eps:
                        H -= p * math.log(p)

                Hmax = math.log(d)
                focus = 1.0 - (H / max(Hmax, eps))
                out[u] += w_k * focus

        return out

    def _dominator_scores(self, sinks: Sequence[Any]) -> Dict[Any, float]:
        """
        Dominator signal: Dom(u)=1 if u dominates ANY sink (w.r.t a super-source).
        We compute dominator sets via an iterative algorithm (portable, works on DAGs).
        """
        roots = self._dominance_roots()
        if not roots:
            return {u: 0.0 for u in self.G.nodes()}

        # Build augmented graph with super-source
        super_source = "__VNS_SUPER_SOURCE__"
        H = self.G.copy()
        if super_source in H:
            # extremely unlikely, but avoid collision
            i = 0
            while f"{super_source}{i}" in H:
                i += 1
            super_source = f"{super_source}{i}"
        H.add_node(super_source)
        for r in roots:
            H.add_edge(super_source, r, **{self.cfg.edge_weight_attr: 0.0})

        # Compute dominator sets for all nodes reachable from super_source
        dom = self._dominators_iterative(H, start=super_source)

        # Union of dominators of sinks
        dom_any: Set[Any] = set()
        for k in sinks:
            if k in dom:
                dom_any |= dom[k]

        out: Dict[Any, float] = {u: (1.0 if u in dom_any else 0.0) for u in self.G.nodes()}
        return out

    def _dominators_iterative(self, H: nx.DiGraph, start: Any) -> Dict[Any, Set[Any]]:
        """
        Classic iterative dominance algorithm:
          dom(start) = {start}
          dom(n) = {n} ∪ ⋂_{p in preds(n)} dom(p)
        """
        reachable = set(nx.descendants(H, start)) | {start}
        nodes = [n for n in H.nodes() if n in reachable]

        dom: Dict[Any, Set[Any]] = {}
        all_nodes = set(nodes)

        for n in nodes:
            dom[n] = {start} if n == start else set(all_nodes)

        changed = True
        while changed:
            changed = False
            for n in nodes:
                if n == start:
                    continue
                preds = [p for p in H.predecessors(n) if p in reachable]
                if not preds:
                    new_dom = {n, start}  # disconnected-ish node from start
                else:
                    intersect = set(dom[preds[0]])
                    for p in preds[1:]:
                        intersect &= dom[p]
                    new_dom = {n} | intersect

                if new_dom != dom[n]:
                    dom[n] = new_dom
                    changed = True

        # Remove the artificial start from returned sets for user-facing nodes if you want;
        # we keep it internal anyway.
        return dom

    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return dense text embeddings using SentenceTransformers if available."""
        docs = [(t or "").strip() for t in texts]
        n_docs = len(docs)
        if n_docs == 0:
            return np.zeros((0, 1), dtype=float)

        normalized = [self._normalize_tag(t) for t in docs]
        model = self._get_text_embedding_model()

        if model is not None:
            model_name = getattr(self.cfg, "text_embedding_model_name", "BAAI/bge-small-en-v1.5").lower()
            formatted: List[str] = []
            for raw in docs:
                cleaned = raw or "[unknown]"
                if "e5" in model_name:
                    formatted.append(f"query: {cleaned}")
                else:
                    formatted.append(cleaned)
            try:
                embeddings = model.encode(
                    formatted,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=min(32, max(1, n_docs)),
                )
                return np.asarray(embeddings, dtype=float)
            except Exception as exc:
                print(
                    f"TopImpactLens: falling back to TF-IDF embeddings because '{model_name}' "
                    f"failed with error: {exc}"
                )

        return self._tfidf_embed_texts(normalized)

    def _get_text_embedding_model(self):
        if SentenceTransformer is None:
            return None
        if self._text_embedding_model is not None:
            return self._text_embedding_model

        model_name = getattr(self.cfg, "text_embedding_model_name", "BAAI/bge-small-en-v1.5")
        try:
            self._text_embedding_model = SentenceTransformer(model_name)
        except Exception as exc:
            print(f"TopImpactLens: unable to load text embedding model '{model_name}': {exc}")
            self._text_embedding_model = None
        return self._text_embedding_model

    def _tfidf_embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Fallback TF-IDF embeddings over token and char n-grams."""
        normalized = [self._normalize_tag(t or "") for t in texts]
        tokenized: List[List[str]] = []
        doc_freq: Counter[str] = Counter()

        for text in normalized:
            features: List[str] = []
            if text:
                word_tokens = re.findall(r"[a-z0-9]+", text)
                features.extend(word_tokens)

                compact = text.replace(" ", "")
                for n in (3, 4):
                    if len(compact) >= n:
                        features.extend(compact[i : i + n] for i in range(len(compact) - n + 1))

            tokenized.append(features)
            if features:
                doc_freq.update(set(features))

        n_docs = len(texts)
        if not doc_freq or n_docs == 0:
            return np.zeros((n_docs, 1), dtype=float)

        vocab = {tok: idx for idx, tok in enumerate(sorted(doc_freq.keys()))}
        idf = {tok: math.log((n_docs + 1) / (doc_freq[tok] + 1)) + 1.0 for tok in vocab}

        embeddings = np.zeros((n_docs, len(vocab)), dtype=float)
        for i, features in enumerate(tokenized):
            if not features:
                continue

            counts = Counter(features)
            total = sum(counts.values())
            if total == 0:
                continue

            for tok, cnt in counts.items():
                j = vocab[tok]
                tf = cnt / total
                embeddings[i, j] = tf * idf[tok]

            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm

        return embeddings

    
    def _normalize_tag(self, tag: str) -> str:
        tag = tag.lower().strip()
        tag = tag.replace("_", " ")
        tag = tag.replace("-", " ")
        return " ".join(tag.split())
    
    def _group_boost_scores(self, importance: Mapping[Any, float]) -> Dict[Any, float]:
        """
        GroupBoost(u) = ClusterMass(tag(u)) / (|cluster|^alpha)
        where tag(u) is a bracket-tag extracted from label like "[harmful requests] ...".
        """
        alpha = self.cfg.group_alpha
        sim_threshold = float(getattr(self.cfg, "group_similarity_threshold", 0.85))
        sim_threshold = max(min(sim_threshold, 1.0), -1.0)

        labeled_nodes: List[Any] = []
        labeled_tags: List[str] = []
        singleton_clusters: List[List[Any]] = []

        for u in self.G.nodes():
            tag = self._bracket_tag(u)
            if tag is None: # ToDo: if a feature is not labeled, we can rerun automated interp and assign a label to it
                singleton_clusters.append([u]) 
            else:
                labeled_nodes.append(u)
                labeled_tags.append(tag)

        clusters: List[List[Any]] = []

        if labeled_nodes:
            embeddings = self._embed_texts(labeled_tags)
            m = len(labeled_nodes)
            parent = list(range(m))

            def find(i: int) -> int:
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            def union(i: int, j: int) -> None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

            sim_matrix = embeddings @ embeddings.T if m > 0 else np.zeros((0, 0))
            for i in range(m):
                for j in range(i + 1, m):
                    if float(sim_matrix[i, j]) >= sim_threshold - 1e-6:
                        union(i, j)

            grouped: Dict[int, List[Any]] = defaultdict(list)
            for idx, node in enumerate(labeled_nodes):
                grouped[find(idx)].append(node)
            clusters.extend(grouped.values())

        clusters.extend(singleton_clusters)

        out: Dict[Any, float] = {}
        eps = self.cfg.eps
        for nodes in clusters:
            if not nodes:
                continue
            mass = sum(float(importance.get(u, 0.0)) for u in nodes)
            denom = (len(nodes) ** alpha) if len(nodes) > 0 else 1.0
            boost = mass / max(denom, eps)
            for u in nodes:
                out[u] = boost
        return out

    # -----------------------------
    # Sink choice + weights
    # -----------------------------
    def _find_logit_nodes(self) -> List[Any]:
        return [u for u in self.G.nodes() if self._node_type_norm(u) == "logit"]

    def _choose_sinks(self, logits: Sequence[Any], target_logit: Optional[Any]) -> Tuple[Any, List[Any]]:
        """
        Choose y* and alts based on node_prob_attr unless target_logit is provided.
        """
        prob_attr = self.cfg.node_prob_attr

        def p(u: Any) -> float:
            return float(self.G.nodes[u].get(prob_attr, 0.0) or 0.0)

        logits_sorted = sorted(logits, key=p, reverse=True)
        if target_logit is not None:
            if target_logit not in self.G:
                raise ValueError("target_logit is not a node in the graph.")
            y_star = target_logit
        else:
            y_star = logits_sorted[0]

        # Choose alternatives
        alts: List[Any] = []
        if self.alt_policy.enabled:
            # Identify top-2 probs (if available) for heuristics
            top_p = p(y_star)
            second_p = p(logits_sorted[1]) if len(logits_sorted) > 1 else 0.0

            trigger = (top_p < self.alt_policy.trigger_if_top_prob_below) or (
                second_p >= self.alt_policy.trigger_if_second_prob_above
            )

            if trigger:
                for u in logits_sorted:
                    if u == y_star:
                        continue
                    if p(u) >= self.alt_policy.min_alt_prob:
                        alts.append(u)
                    if len(alts) >= self.alt_policy.max_alts:
                        break

        return y_star, alts

    def _sink_weights(self, y_star: Any, alts: Sequence[Any]) -> Dict[Any, float]:
        """
        omega_y* = 1
        omega_alt = lambda_alt * p(alt) / sum p(alt)
        """
        prob_attr = self.cfg.node_prob_attr

        def p(u: Any) -> float:
            return float(self.G.nodes[u].get(prob_attr, 0.0) or 0.0)

        omega: Dict[Any, float] = {y_star: 1.0}
        if not alts:
            return omega

        s = sum(max(p(a), 0.0) for a in alts)
        if s <= self.cfg.eps:
            # fallback: uniform among alts
            for a in alts:
                omega[a] = self.cfg.lambda_alt / len(alts)
            return omega

        for a in alts:
            omega[a] = self.cfg.lambda_alt * (max(p(a), 0.0) / s)
        return omega

    # -----------------------------
    # Priors + parsing helpers
    # -----------------------------
    def _node_type_norm(self, u: Any) -> str:
        """
        Normalize node feature_type strings into {logit, embedding, error, unexplored, feature, other}.
        """
        raw = str(self.G.nodes[u].get(self.cfg.node_type_attr, "") or "").strip().lower()
        if raw == "logit":
            return "logit"
        if raw == "embedding":
            return "embedding"
        if "error" in raw:
            return "error"
        if "unexplored" in raw:
            return "unexplored"
        # treat cross-layer transcoder as 'feature'
        if "transcoder" in raw or "feature" in raw:
            return "feature"
        return "other"

    def _label(self, u: Any) -> str:
        return str(self.G.nodes[u].get(self.cfg.node_label_attr, "") or "")

    def _bracket_tag(self, u: Any) -> Optional[str]:
        """
        Extract bracket tag like "[harmful requests]" from label, if present.
        """
        label = self._label(u).strip()
        if not label:
            return None
        
        # m = re.match(self.cfg.bracket_tag_regex, label)
        m = label.strip() # take the whole label as tag
        
        if not m:
            return None
        
        # tag = m.group(1).strip()
        tag = m
        tag = self._normalize_tag(tag)
        
        return tag or None

    def _embedding_token(self, u: Any) -> Optional[str]:
        """
        If label looks like 'Emb: " ... "', extract token string.
        """
        label = self._label(u)
        if "Emb:" not in label:
            return None
        m = re.search(r'Emb:\s*"(.*)"', label)
        return m.group(1) if m else None

    def _prior(self, u: Any) -> float:
        """
        prior(u) = type_prior * label_prior * (optional embedding formatting prior)
        """
        t = self._node_type_norm(u)
        p_type = float(self.priors.type_prior.get(t, self.priors.type_prior.get("other", 0.6)))

        label = self._label(u).strip()
        if not label:
            p_label = self.priors.empty_label_prior
        else:
            if re.search(self.cfg.assistant_label_regex, label.lower()):
                p_label = self.priors.assistant_label_prior
            else:
                p_label = 1.0

        p_embed = 1.0
        if t == "embedding":
            tok = (self._embedding_token(u) or "").strip()
            # very lightweight formatting heuristic
            if tok in {"", "⏎⏎", "⏎", ":", ",", ".", "(", ")", ")))", "(((", "((", "))"}:
                p_embed = self.priors.formatting_embedding_prior

        return max(p_type * p_label * p_embed, 0.0)

    # -----------------------------
    # Dominance root selection
    # -----------------------------
    def _dominance_roots(self) -> List[Any]:
        """
        Roots used for dominance: prefer embeddings with out_degree>0;
        otherwise fallback to in_degree==0 nodes.
        """
        if self.cfg.dominance_prefer_embeddings:
            embs = [u for u in self.G.nodes() if self._node_type_norm(u) == "embedding" and self.G.out_degree(u) > 0]
            if embs:
                return embs

        # fallback: structural sources
        roots = [u for u in self.G.nodes() if self.G.in_degree(u) == 0 and self.G.out_degree(u) > 0]
        return roots
