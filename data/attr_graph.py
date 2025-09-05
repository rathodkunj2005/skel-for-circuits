import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from circuit_tracer.graph import prune_graph

def graph_prunings(
    graph,
    node_thresholds,
    edge_threshold = 1.0
):
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

def metric_fn(logits):
    logits = logits.squeeze()[-1]
    answer_logit = logits[answer_idx]
    top_values, top_indicies = torch.topk(logits, 11)
    mask = top_indicies != answer_idx
    mean_top_10 = top_values[mask][:10].mean()
    return answer_logit - mean_top_10


def get_first_predicted_token(model_name, prompt):
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


def get_graph(model, prompt, max_n_logits, desired_logit_prob, graph_batch_size, max_feature_nodes, offload='cpu', verbose=True):
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

    answer, answer_idx = get_first_predicted_token(model_name, prompt)

    with torch.inference_mode():
        full_logits = model(prompt)
        full_metric = metric_fn(full_logits).item()
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
        empty_metric = metric_fn(empty_logits).item()
        

    node_thresholds = np.arange(0.05, 1.0, 0.01)
    pruned_graphs = graph_prunings(
        graph=graph,
        node_thresholds=node_thresholds,
    )

    return pruned_graphs[1]