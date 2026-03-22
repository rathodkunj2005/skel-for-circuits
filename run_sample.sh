#!/usr/bin/env bash
# Quick-start: run LLM grouping on the bundled sample graph via Ollama.
# Usage:  bash run_sample.sh [graph_json_path]
set -e

GRAPH="${1:-data/sample_graphs/Gemma/gemma-fact-dallas-austin.json}"

USE_OLLAMA=true python3 -m experiments.run_experiment \
    --graph "$GRAPH" \
    --grouping llm \
    --open_source_grouping \
    --open_source_labeling
