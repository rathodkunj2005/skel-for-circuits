import sys
from pathlib import Path

# Add submodule path to Python import path
submodule_path = Path(__file__).resolve().parent.parent / "external" / "Faithfulness"
sys.path.append(str(submodule_path))

import json
import time
import re
import numpy as np


def _import_circuit_tracer():
    """Lazily import circuit_tracer (only needed for --prompt mode)."""
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from circuit_tracer.graph import prune_graph
    from circuit_tracer import attribute
    from circuit_tracer.utils.create_graph_files import (
        build_model,
        create_nodes,
        create_used_nodes_and_edges,
    )
    return prune_graph, attribute, build_model, create_nodes, create_used_nodes_and_edges


TLENS_MODEL_ID_TO_NP_MODEL_ID = {
    "google/gemma-2-2b": "gemma-2-2b",
    "meta-llama/Llama-3.2-1B": "llama3.1-8b",
    "Qwen/Qwen3-4B": "qwen3-4b",
}


def graph_prunings(graph, node_thresholds, edge_threshold=1.0):
    import torch
    from tqdm import tqdm
    prune_graph, _, _, _, _ = _import_circuit_tracer()
    results = []

    n_features = len(graph.selected_features)
    n_tokens = graph.n_pos
    n_error_nodes = graph.cfg.n_layers * n_tokens

    for threshold in tqdm(node_thresholds):

        prune_result = prune_graph(graph, node_threshold=threshold, edge_threshold=edge_threshold)
        node_mask = prune_result.node_mask

        feature_mask = node_mask[:n_features]
        selected_feature_indices = torch.where(feature_mask)[0]
        active_feature_indices = graph.selected_features[selected_feature_indices]
        feature_nodes = graph.active_features[active_feature_indices].tolist()
        
        error_mask = node_mask[n_features : n_features + n_error_nodes]
        error_indices = torch.where(error_mask)[0]
        error_nodes = []

        for flat_idx in error_indices:
            layer = flat_idx.item() // n_tokens
            pos = flat_idx.item() % n_tokens
            error_nodes.append((layer, pos))
            
        result_entry = {
            'feature_nodes': feature_nodes,
            'error_nodes': error_nodes,
        }
        results.append(result_entry)

    return results


def metric_fn(logits, answer_idx):
    import torch
    logits = logits.squeeze()[-1]
    answer_logit = logits[answer_idx]
    top_values, top_indicies = torch.topk(logits, 11)
    mask = top_indicies != answer_idx
    mean_top_10 = top_values[mask][:10].mean()
    return answer_logit - mean_top_10


def get_first_predicted_token(model_name, prompt):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    next_token_logits = logits[0, -1, :]
    predicted_token_id = torch.argmax(next_token_logits).item()
    
    predicted_token = tokenizer.decode(predicted_token_id)
    
    return predicted_token, predicted_token_id


def connected_graph_prunings(model, model_name, prompt, graph, node_threshold, edge_threshold, max_n_logits, desired_logit_prob, batch_size, max_feature_nodes):
    import torch
    from transformers import AutoTokenizer
    prune_graph, _, build_model, create_nodes, create_used_nodes_and_edges = _import_circuit_tracer()
    current_time_ms = int(time.time() * 1000)
    
    cleaned_string = re.sub(r'[^a-zA-Z0-9]', '', prompt).lower().replace(" ", "")
    slug_identifier = cleaned_string + "-" + str(int(current_time_ms/1000))
    
    pruned_graph = prune_graph(graph, node_threshold, edge_threshold)
    
    node_mask, edge_mask, cumulative_scores = (
        el.cpu() for el in pruned_graph
    )

    tokenizer = AutoTokenizer.from_pretrained(model.cfg.tokenizer_name)
    nodes = create_nodes(
        graph,
        node_mask,
        tokenizer,
        cumulative_scores,
    )
    print("nodes created")

    used_nodes, used_edges = create_used_nodes_and_edges(
        graph, nodes, edge_mask
    )
    print("used nodes and edges created")

    output_model = build_model(
        graph,
        used_nodes,
        used_edges,
        slug_identifier,
        TLENS_MODEL_ID_TO_NP_MODEL_ID[model_name],
        node_threshold,
        tokenizer,
    )
    print("output model created")

    model_dict = output_model.model_dump()

    model_dict["metadata"]["info"] = {
        "creator_name": "fatemehashemi",
        # "creator_url": "https://neuronpedia.org",
        "creator_url": "https://github.com/safety-research/circuit-tracer",
        "source_urls": [
            "https://neuronpedia.org/gemma-2-2b/gemmascope-transcoder-16k",
            "https://huggingface.co/google/gemma-scope-2b-pt-transcoders"
        ],
        # "transcoder_set": transcoder_set,
        "generator": {
            "name": "circuit-tracer",
            "version": "1.0.0",
            "url": "https://github.com/safety-research/circuit-tracer",
        },
        "create_time_ms": current_time_ms,
        "neuronpedia_link": "",
        "neuronpedia_source_set": "https://www.neuronpedia.org/gemma-2-2b/gemmascope-transcoder-16k"
    }
    
    model_dict["metadata"]["schema_version"] = "1.0"
    
    model_dict["metadata"]["generation_settings"] = {
        "max_n_logits": max_n_logits,
        "desired_logit_prob": desired_logit_prob,
        "batch_size": batch_size,
        "max_feature_nodes": max_feature_nodes,
    }
    
    # model_dict["metadata"]["nodeThreshold"] = node_threshold

    model_dict["metadata"]["pruning_settings"] = {
        "node_threshold": node_threshold,
        "edge_threshold": edge_threshold,
    }
    
    model_json = json.dumps(model_dict)
    # print(model_json)
    return model_json

    
def get_attribution_graph(model, model_name, prompt, max_n_logits, desired_logit_prob, graph_batch_size, max_feature_nodes, node_threshold, edge_threshold, offload='cpu', verbose=True):
    _, attribute, _, _, _ = _import_circuit_tracer()
    offload = 'cpu'
    verbose = True

    graph = attribute(
        prompt=prompt,
        model=model,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
        batch_size=graph_batch_size,
        max_feature_nodes=max_feature_nodes,
        offload=offload,
        verbose=verbose
    )
    graph = connected_graph_prunings(model, model_name, prompt, graph, node_threshold, edge_threshold, max_n_logits, desired_logit_prob, graph_batch_size, max_feature_nodes)
    

    answer, answer_idx = get_first_predicted_token(model_name, prompt)
    return graph, answer, answer_idx

    with torch.inference_mode():
        full_logits = model(prompt)
        full_metric = metric_fn(full_logits, answer_idx).item()
        empty_logits = model.graph_ablation(
            input=prompt,
            selected_features=[],
            selected_errors=[],
            mean_ablate=False,
            mean_ablation_samples=1000,
            retain_bos_features=True,
            direct_effects=True,
            freeze_attention=True
        )
        empty_metric = metric_fn(empty_logits, answer_idx).item()
        

    # node_thresholds = np.arange(0.05, 1.0, 0.01)
    node_thresholds = [0.5]

    pruned_graphs = graph_prunings(
        graph=graph,
        node_thresholds=node_thresholds,
    )

    return pruned_graphs