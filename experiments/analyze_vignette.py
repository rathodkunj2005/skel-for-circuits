"""
analyze_vignette.py — End-to-end circuit analysis pipeline.

Usage:
    python experiments/analyze_vignette.py --vignette data/outputs/Gemma/gemma-fact-dallas-austin_skeleton.json
    python experiments/analyze_vignette.py --vignette path/to/skeleton.json --model gpt-4o
    python experiments/analyze_vignette.py --vignette path/to/skeleton.json --counterfactual "base-11 addition"

Reads a vignette/skeleton JSON, sends it to an analysis LLM with a structured
prompt, and writes a full markdown report to data/analysis_reports/.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in mechanistic interpretability of large language models. \
You analyze sparse feature circuits (SFCs) — computational subgraphs extracted \
from a language model that are causally responsible for a specific prediction. \
Your analysis is precise, grounded in the circuit data, and avoids speculation \
beyond what the circuit supports.\
"""

ANALYSIS_PROMPT_TEMPLATE = """\
You are given a sparse feature circuit extracted from a language model in JSON format.

## Task context
- **Test model (whose circuit this is):** {scan}
- **Input prompt the circuit was extracted from:** `{prompt}`
- **Default task description:** {task_description}
- **Counterfactual task to predict generalization for:** {counterfactual}

## Circuit summary
The circuit has been skeletonized into supernodes (semantic clusters of features).
Supernodes (each is a semantically coherent group of nodes):
{supernode_summary}

Key pinned nodes (highest-influence nodes selected by the skeletonization pipeline):
{top_nodes_summary}

## Full circuit JSON
```json
{circuit_json}
```

## Your analysis tasks

1. **Semantic clusters** — Identify the main semantic clusters and what concept each represents.
2. **Computational pathway** — Trace the main activation path from input tokens to the output token.
3. **High-influence & bottleneck nodes** — Which nodes/supernodes are most critical? Are there bottlenecks?
4. **Circuit function** — Is this factual recall, arithmetic reasoning, pattern matching, or something else?
5. **Reasoning vs. memorization** — Does this circuit reflect robust algorithmic reasoning or surface-level memorization? Cite specific nodes.
6. **Generalization analysis** — Which circuit components are task-general vs. task-specific to the default task? Would those components transfer to the counterfactual task?

## Final verdict (REQUIRED — answer this last, after your analysis)

Based solely on what this circuit reveals about the model's internal computation, predict whether the test model will correctly answer instances of the counterfactual task.

End your response with exactly this format:
```
GENERALIZATION VERDICT: [YES or NO]
CONFIDENCE: [integer 0-100]%
KEY EVIDENCE: [one sentence citing the specific circuit component(s) that drive this prediction]
```
"""


def build_supernode_summary(qparams: dict) -> str:
    supernodes = qparams.get("supernodes", [])
    if not supernodes:
        return "(no supernodes found)"
    lines = []
    for sn in supernodes:
        label = sn[0]
        members = sn[1:]
        lines.append(f"  - **{label}** → nodes: {', '.join(members)}")
    return "\n".join(lines)


def _get_priority_node_ids(qparams: dict) -> list[str]:
    """
    Return node IDs in priority order: pinnedIds first (skeletonizer chose these
    as the highest-influence nodes), then any extra IDs found in supernodes.
    """
    pinned = list(qparams.get("pinnedIds", []))
    pinned_set = set(pinned)
    # Add supernode members not already in pinnedIds
    for sn in qparams.get("supernodes", []):
        for nid in sn[1:]:
            if nid not in pinned_set:
                pinned.append(nid)
                pinned_set.add(nid)
    return pinned


def build_top_nodes_summary(nodes: list, qparams: dict, top_k: int = 15) -> str:
    """
    Return top_k nodes, prioritizing pinnedIds from qParams (the nodes the
    skeletonizer judged most influential). Falls back to influence×activation
    sort for any remaining slots.
    """
    node_by_id = {n["node_id"]: n for n in nodes}
    priority_ids = _get_priority_node_ids(qparams)

    selected = []
    seen = set()
    # Priority: pinned/supernode members first
    for nid in priority_ids:
        if nid in node_by_id and nid not in seen:
            selected.append(node_by_id[nid])
            seen.add(nid)
        if len(selected) >= top_k:
            break

    # Fill remaining slots by influence*activation if needed
    if len(selected) < top_k:
        scored = []
        for n in nodes:
            if n["node_id"] in seen:
                continue
            influence = n.get("influence", 0) or 0
            activation = n.get("activation", 0) or 0
            scored.append((abs(influence) * abs(activation), n))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, n in scored:
            if len(selected) >= top_k:
                break
            selected.append(n)
            seen.add(n["node_id"])

    lines = []
    for n in selected:
        label = n.get("clerp") or n.get("label") or "[unlabeled]"
        node_id = n.get("node_id", "?")
        layer = n.get("layer", "?")
        influence = n.get("influence", 0) or 0
        activation = n.get("activation", 0) or 0
        lines.append(
            f"  - `{node_id}` (layer {layer}) | influence={influence:.3f} "
            f"activation={activation:.3f} | \"{label}\""
        )
    return "\n".join(lines) if lines else "(no nodes found)"


def build_circuit_json_for_prompt(data: dict, max_nodes: int = 60) -> str:
    """
    Build a compact JSON representation for LLM consumption.
    Prioritizes pinnedIds/supernode members (the skeletonizer's important nodes),
    then includes their inter-links.
    """
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    qparams = data.get("qParams", {})

    node_by_id = {n["node_id"]: n for n in nodes}
    priority_ids = _get_priority_node_ids(qparams)

    # Build selected node list: priority nodes first, then fill by influence*activation
    selected_ids = []
    seen = set()
    for nid in priority_ids:
        if nid in node_by_id and nid not in seen:
            selected_ids.append(nid)
            seen.add(nid)
        if len(selected_ids) >= max_nodes:
            break

    if len(selected_ids) < max_nodes:
        scored = sorted(
            [(abs(n.get("influence", 0) or 0) * abs(n.get("activation", 0) or 0), n)
             for n in nodes if n["node_id"] not in seen],
            key=lambda x: x[0], reverse=True
        )
        for _, n in scored:
            if len(selected_ids) >= max_nodes:
                break
            selected_ids.append(n["node_id"])
            seen.add(n["node_id"])

    top_ids = set(selected_ids)
    top_nodes = [node_by_id[nid] for nid in selected_ids if nid in node_by_id]

    # Links between selected nodes, sorted by weight
    filtered_links = [
        lnk for lnk in links
        if lnk.get("source") in top_ids and lnk.get("target") in top_ids
    ]
    filtered_links.sort(key=lambda x: abs(x.get("weight", 0)), reverse=True)
    filtered_links = filtered_links[: max_nodes * 3]

    compact = {
        "metadata": {
            "scan": data.get("metadata", {}).get("scan", ""),
            "prompt": data.get("metadata", {}).get("prompt", ""),
            "slug": data.get("metadata", {}).get("slug", ""),
        },
        "supernodes": qparams.get("supernodes", []),
        "nodes": [
            {
                "node_id": n.get("node_id"),
                "layer": n.get("layer"),
                "feature_type": n.get("feature_type"),
                "clerp": n.get("clerp") or n.get("label"),
                "influence": round(n.get("influence", 0) or 0, 4),
                "activation": round(n.get("activation", 0) or 0, 4),
                "pinned": n.get("node_id") in set(qparams.get("pinnedIds", [])),
            }
            for n in top_nodes
        ],
        "links": [
            {
                "source": lnk["source"],
                "target": lnk["target"],
                "weight": round(lnk.get("weight", 0), 4),
            }
            for lnk in filtered_links
        ],
    }
    return json.dumps(compact, indent=2)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_analysis_llm(prompt: str, model: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=4096,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(
    output_path: str,
    vignette_path: str,
    metadata: dict,
    full_prompt: str,
    llm_response: str,
    model: str,
    task_description: str,
    counterfactual: str,
    elapsed_s: float,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    slug = metadata.get("slug", Path(vignette_path).stem)
    scan = metadata.get("scan", "unknown")
    circuit_prompt = metadata.get("prompt", "")

    lines = [
        f"# Circuit Analysis Report",
        f"",
        f"**Generated:** {timestamp}  ",
        f"**Vignette:** `{vignette_path}`  ",
        f"**Model (scan):** `{scan}`  ",
        f"**Analysis LLM:** `{model}`  ",
        f"**Slug:** `{slug}`  ",
        f"**Elapsed:** {elapsed_s:.1f}s  ",
        f"",
        f"---",
        f"",
        f"## Circuit Prompt",
        f"",
        f"```",
        circuit_prompt,
        f"```",
        f"",
        f"## Task Description",
        f"",
        task_description,
        f"",
        f"## Counterfactual Task",
        f"",
        counterfactual if counterfactual else "_None specified_",
        f"",
        f"---",
        f"",
        f"## Full Analysis Prompt Sent to LLM",
        f"",
        f"<details>",
        f"<summary>Click to expand full prompt</summary>",
        f"",
        f"```",
        full_prompt,
        f"```",
        f"",
        f"</details>",
        f"",
        f"---",
        f"",
        f"## LLM Analysis",
        f"",
        llm_response,
        f"",
        f"---",
        f"",
        f"_Report generated by `experiments/analyze_vignette.py`_",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a vignette/skeleton JSON with an LLM and produce a markdown report."
    )
    parser.add_argument(
        "--vignette", required=True,
        help="Path to skeleton JSON file (output of run_experiment.py)",
    )
    parser.add_argument(
        "--model", default="gpt-5.4",
        help="OpenAI model to use for analysis (default: gpt-5.4)",
    )
    parser.add_argument(
        "--task", default="",
        help="Human-readable description of the default task (e.g. 'base-10 addition')",
    )
    parser.add_argument(
        "--counterfactual", default="",
        help="Description of the counterfactual task to predict generalization for",
    )
    parser.add_argument(
        "--max_nodes", type=int, default=60,
        help="Max nodes to include in the JSON sent to the LLM (default: 60)",
    )
    parser.add_argument(
        "--output_dir", default="data/analysis_reports",
        help="Directory to write markdown reports (default: data/analysis_reports)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY environment variable not set.")

    # Load vignette
    vignette_path = args.vignette
    print(f"Loading vignette: {vignette_path}")
    with open(vignette_path) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    qparams = data.get("qParams", {})
    nodes = data.get("nodes", [])

    scan = metadata.get("scan", "unknown model")
    prompt_text = metadata.get("prompt", "")
    task_description = args.task or f"Factual/reasoning task: '{prompt_text}'"
    counterfactual = args.counterfactual or "Not specified"

    # Build prompt components
    supernode_summary = build_supernode_summary(qparams)
    top_nodes_summary = build_top_nodes_summary(nodes, qparams)
    circuit_json = build_circuit_json_for_prompt(data, max_nodes=args.max_nodes)

    full_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        scan=scan,
        prompt=prompt_text,
        task_description=task_description,
        counterfactual=counterfactual,
        supernode_summary=supernode_summary,
        top_nodes_summary=top_nodes_summary,
        circuit_json=circuit_json,
    )

    # Call LLM
    print(f"Calling {args.model} for circuit analysis...")
    import time
    t0 = time.time()
    llm_response = call_analysis_llm(full_prompt, model=args.model, api_key=api_key)
    elapsed = time.time() - t0
    print(f"Done ({elapsed:.1f}s).")

    # Save report
    slug = metadata.get("slug", Path(vignette_path).stem)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{slug}_{timestamp_str}.md"
    output_path = os.path.join(args.output_dir, report_filename)

    write_report(
        output_path=output_path,
        vignette_path=vignette_path,
        metadata=metadata,
        full_prompt=full_prompt,
        llm_response=llm_response,
        model=args.model,
        task_description=task_description,
        counterfactual=counterfactual,
        elapsed_s=elapsed,
    )


if __name__ == "__main__":
    main()



'''  Usage                                                                                                
                                                                                                     
  # Basic
  python experiments/analyze_vignette.py \
    --vignette data/outputs/Gemma/gemma-fact-dallas-austin_skeleton.json                               
                                                                                                       
  # With task context and counterfactual                                                               
  python experiments/analyze_vignette.py \                                                             
    --vignette data/outputs/Gemma/gemma-fact-dallas-austin_skeleton.json \                             
    --task "capital city factual recall" \                                                           
    --counterfactual "base-11 arithmetic addition" \                                                   
    --model gpt-4o                                                                                     
                                                                                                       
  # Control how many nodes go to the LLM (default 60)                                                  
  python experiments/analyze_vignette.py \                                                           
    --vignette path/to/skeleton.json \                                                                 
    --max_nodes 80                                                                                   

  Requires OPENAI_API_KEY in your environment (or .env file). Each run produces a new file like        
  data/analysis_reports/gemma-fact-dallas-austin_20260405_143022.md.'''