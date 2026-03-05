import re
import os
import json
import argparse
import warnings
from typing import List, Tuple, Dict, Any
from pathlib import Path

import numpy as np

from data.loaders import (
    load_graph, load_all_graphs, save_graph_with_qparams,
    send_subgraph_to_api, get_top_logit_node,
)
from data.attr_graph import get_attribution_graph
from lenses.supernode import SupernodeLens
from mapper.top_impact import TopImpactScorer
from mapper.pipeline import MapperPipeline
from mapper.llm_grouping import LLMGroupingPipeline

from data.supernode_label import rerun_auto_interpretation

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------
# Builders
# ----------------------------
def build_lenses(use_closed_source_labeling: bool = True):
    """lenses[0] = SupernodeLens (semantic), lenses[1] = TopImpactScorer (node selection)."""
    return [SupernodeLens(use_closed_source_labeling=use_closed_source_labeling), TopImpactScorer()]


def build_pipeline(args, lenses):
    """Return the appropriate pipeline based on --grouping."""
    if args.grouping == "llm":
        print(
            f"---> Using LLM-based grouping "
            f"(closed_source_grouping={not args.open_source_grouping}, "
            f"closed_source_labeling={not args.open_source_labeling}, "
            f"n_groups_hint={args.n_groups_hint})"
        )
        return LLMGroupingPipeline(
            lenses,
            use_closed_source=not args.open_source_grouping,
            use_closed_source_labeling=not args.open_source_labeling,
            n_groups_hint=args.n_groups_hint if args.n_groups_hint > 0 else None,
        )
    print("---> Using embedding-based grouping (MapperPipeline)")
    return MapperPipeline(lenses)



# ----------------------------
# Graph loading / building
# ----------------------------
def get_json_files_from_directory(directory: str) -> List[str]:
    """Return sorted list of .json file paths in directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    json_files = sorted([str(f) for f in dir_path.glob("*.json")])
    if not json_files:
        raise ValueError(f"No JSON files found in directory: {directory}")

    return json_files


def load_graph_info_from_file(graph_path: str) -> Tuple[Any, str, str, Dict[str, Any]]:
    """Load graph and extract prompt, answer, metadata from JSON file."""
    graph = load_graph(graph_path)
    file_data = json.load(open(graph_path, "r"))

    prompt = file_data.get("metadata", {}).get("prompt", "")
    answer_node = get_top_logit_node(graph)
    print(graph_path)
    try:
        answer = graph.nodes[answer_node]["clerp"].split('"')[1]
    except Exception:
        answer = "none"
        print("Main Run | not typical logit clerp:", answer_node)
    metadata = file_data.get("metadata", {})

    return graph, prompt, answer, metadata


# ----------------------------
# Graph loading / building
# ----------------------------
def load_test_graphs_and_answer(args, repl_model, model_name):
    """Return (test_graphs, prompts, answers, metadatas).
    
    - If --graph_dir provided: load all JSONs from directory
    - If --graph provided: load single graph file
    - Otherwise: build attribution graph from prompt
    """
    print("Loading test graphs...")

    if args.graph_dir:
        json_files = get_json_files_from_directory(args.graph_dir)
        print(f"Found {len(json_files)} JSON files in {args.graph_dir}")

        test_graphs, prompts, answers, metadatas = [], [], [], []
        for json_file in json_files:
            graph, prompt, answer, metadata = load_graph_info_from_file(json_file)
            test_graphs.append(graph)
            prompts.append(prompt)
            answers.append(answer)
            metadatas.append(metadata)

        return test_graphs, prompts, answers, metadatas

    if args.graph != "":
        graph, prompt, answer, metadata = load_graph_info_from_file(args.graph)
        return [graph], [prompt], [answer], [metadata]

    # Build attribution graph from prompt
    pruned_g, answer, answer_idx = get_attribution_graph(
        repl_model,
        model_name,
        args.prompt,
        args.max_n_logits,
        args.desired_logit_prob,
        args.graph_batch_size,
        args.max_feature_nodes,
        args.node_threshold,
        args.edge_threshold,
    )
    pruned_g = json.loads(pruned_g)
    metadata = pruned_g["metadata"]
    test_graphs = [load_graph(pruned_g)]
    return test_graphs, [args.prompt], [answer], [metadata]


# ----------------------------
# Output helpers
# ----------------------------
def output_paths(args, base_name: str) -> Tuple[str, str]:
    """Return (output_base_dir, output_path)."""
    if args.graph_dir:
        output_base_dir = os.path.join("data/outputs/", Path(args.graph_dir).name)
    elif "Haiku" in (args.graph or ""):
        output_base_dir = "data/outputs/Haiku/"
    else:
        output_base_dir = "data/outputs/Gemma/"

    os.makedirs(output_base_dir, exist_ok=True)
    output_path = os.path.join(output_base_dir, f"{base_name}_skeleton.json")
    return output_base_dir, output_path


def process_single_graph(i: int, total: int, graph, args, lenses, pipeline,
                         prompt: str, answer: str, metadata: Dict[str, Any],
                         graph_source: str = "",
                         use_closed_source_labeling: bool = True):
    """Process a single graph and save skeleton to disk."""
    print(f"\n{'='*60}")
    print(f"Processing graph {i+1}/{total}")
    print(f"{'='*60}")

    if graph_source:
        base_name = os.path.splitext(os.path.basename(graph_source))[0]
    elif args.graph != "":
        base_name = os.path.splitext(os.path.basename(args.graph))[0]
    else:
        base_name = re.sub(r"[^a-zA-Z0-9]", "", prompt).lower().replace(" ", "")

    print("GRAPH:", base_name)

    skeleton = pipeline(graph)

    output_base_dir, output_path = output_paths(args, base_name)

    source_file = graph_source if graph_source else args.graph
    if source_file:
        file_meta = json.load(open(source_file))["metadata"]
        outcome_data = save_graph_with_qparams(skeleton, graph, file_meta, output_path,
                                               use_closed_source_labeling=use_closed_source_labeling)
        send_subgraph_to_api(outcome_data, displayName="abs-tresh-035-llm")
    else:
        outcome_data = save_graph_with_qparams(skeleton, graph, metadata, output_path,
                                               use_closed_source_labeling=use_closed_source_labeling)

    print(f"Saved skeleton to {output_path}")
    return output_base_dir


# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply mapper skeletonization to an attribution graph"
    )
    parser.add_argument("--prompt", required=False, type=str, help="input prompt")
    parser.add_argument("--desired_logit_prob", type=float, default=0.95)
    parser.add_argument("--max_n_logits", type=int, default=10)
    parser.add_argument("--max_feature_nodes", type=int, default=8192)
    parser.add_argument("--graph_batch_size", type=int, default=256)
    parser.add_argument("--node_threshold", type=float, default=0.5)
    parser.add_argument("--edge_threshold", type=float, default=0.8)
    parser.add_argument("--graph", required=False, default="", help="Path to input graph JSON")
    parser.add_argument("--graph_dir", required=False, default="",
                        help="Path to directory containing multiple graph JSON files")
    parser.add_argument("--grouping", choices=["embedding", "llm"], default="embedding",
                        help="Grouping method: 'embedding' (PCA-based, default) or 'llm' (LLM-based)")
    parser.add_argument("--open_source_grouping", action="store_true", default=False,
                        help="With --grouping llm: use local open-source model for LLM-based grouping instead of OpenAI API")
    parser.add_argument("--open_source_labeling", action="store_true", default=False,
                        help="Use local open-source model for rerun_auto_interpretation and get_umbrella_term instead of OpenAI API")
    parser.add_argument("--n_groups_hint", type=int, default=0,
                        help="With --grouping llm: optional soft hint for number of groups (0 = no hint)")

    args = parser.parse_args()

    input_count = sum([
        bool(args.graph),
        bool(args.graph_dir),
        bool(args.prompt and args.prompt.strip()),
    ])

    if input_count == 0:
        parser.error("Provide one of: --graph, --graph_dir, or --prompt")
    if input_count > 1:
        parser.error("Provide only ONE of: --graph, --graph_dir, or --prompt")

    return args


def main():
    args = parse_args()
    lenses = build_lenses(use_closed_source_labeling=not args.open_source_labeling)

    # Only load the replacement model when building a graph from a prompt
    repl_model = None
    model_name = "google/gemma-2-2b"
    if args.prompt:
        import torch
        from circuit_tracer import ReplacementModel
        print("---> Load replacement model")
        repl_model = ReplacementModel.from_pretrained(
            model_name, "gemma", device="cuda", dtype=torch.bfloat16
        )

    pipeline = build_pipeline(args, lenses)

    print("---> Skeletonizing...")
    test_graphs, prompts, answers, metadatas = load_test_graphs_and_answer(args, repl_model, model_name)
    print(f"Loaded {len(test_graphs)} test graphs")

    graph_sources = get_json_files_from_directory(args.graph_dir) if args.graph_dir else []

    last_output_dir = None
    for i, (graph, prompt, answer, metadata) in enumerate(zip(test_graphs, prompts, answers, metadatas)):
        graph_source = graph_sources[i] if graph_sources else ""
        last_output_dir = process_single_graph(
            i=i, total=len(test_graphs), graph=graph, args=args, lenses=lenses, pipeline=pipeline,
            prompt=prompt, answer=answer, metadata=metadata, graph_source=graph_source,
            use_closed_source_labeling=not args.open_source_labeling,
        )

    print("\n" + "=" * 60)
    print("=== PROCESS COMPLETE ===")
    print("=" * 60)
    print(f"Generated {len(test_graphs)} skeletons in {last_output_dir}")


if __name__ == "__main__":
    main()

