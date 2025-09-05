import json
import os
import networkx as nx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_first_predicted_token(model_name, prompt):
    """
    Get the first token predicted by an LLM given a prompt.
    
    Args:
        model_name (str): Name of the Hugging Face model (e.g., 'gpt2', 'microsoft/DialoGPT-medium')
        prompt (str): Input text prompt
        
    Returns:
        str: The first predicted token as a string
    """
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


def load_graph_from_prompt(model, prompt, first_pred_tok):
    pass


def load_graph(file_path):
    """Load graph from JSON file and convert to NetworkX DiGraph."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes
    for node in data['nodes']:
        G.add_node(
                node['node_id'], 
                feature=node['feature'], 
                label=node.get('clerp', 'blank'),
                clerp=node.get('clerp', 'blank'),
                layer=node['layer'],
                ctx_idx=node['ctx_idx'],
                feature_type=node['feature_type'],
                token_prob=node['token_prob'],
                is_target_logit=node['is_target_logit'],
                run_idx=node['run_idx'],
                reverse_ctx_idx=node['reverse_ctx_idx'],
                jsNodeId=node['jsNodeId'],
                influence=node['influence'],
                activation=node['activation']
            )
            
    # Add edges
    for link in data['links']:
        G.add_edge(link['source'], link['target'], weight=link['weight'])
    
    print(f"---> Loaded the input graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")

    return G


def load_all_graphs(directory):
    """Load all JSON graphs from a directory."""
    graphs = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            graphs.append(load_graph(filepath))
    return graphs


def get_output_nodes(G):
    """Identify output nodes based on node_id pattern."""
    output_nodes = []
    max_prefix = -1
    
    # Find max prefix
    for node in G.nodes():
        try:
            prefix = int(node.split('_')[0])
        except:
            prefix = -1
        if prefix > max_prefix:
            max_prefix = prefix
    
    # Add nodes with max prefix
    for node in G.nodes():
        try:
            prefix = int(node.split('_')[0])
        except:
            prefix = -1
        if prefix == max_prefix:
            output_nodes.append(node)

    return output_nodes


def get_input_nodes(G):
    """Identify input nodes based on node_id pattern."""
    input_nodes = []
    
    for node in G.nodes():
        try:
            prefix = node.split('_')[0]
        except:
            prefix = -1
        if prefix == "E":
            input_nodes.append(node)
    
    return input_nodes


def extract_all_features_and_errors(G):
    """Extract selected_features and selected_errors from the graph.
    
    - selected_features: List[(layer, ctx_idx, feature)] for nodes where feature_type == "cross layer transcoder".
    - selected_errors: List[(layer, ctx_idx)] for nodes where feature_type == "mlp reconstruction error".
    """
    selected_features = []
    selected_errors = []

    for node_id, attrs in G.nodes(data=True):
        feature_type = attrs.get("feature_type")
        layer = attrs.get("layer")
        ctx_idx = attrs.get("ctx_idx")

        if feature_type == "cross layer transcoder":
            feature = attrs.get("feature")
            selected_features.append((layer, ctx_idx, feature))
        elif feature_type == "mlp reconstruction error":
            selected_errors.append((layer, ctx_idx))

    g_dict = {
        "feature_nodes": selected_features,
        "error_nodes": selected_errors
    }
    return g_dict


def extract_per_layer_features_and_errors(G):
    """Extract selected_features and selected_errors from the graph, grouped by layer.
    
    Returns:
        List[Dict]: A list where index == layer number.
        Each element is a dict with:
            - 'feature_nodes': [(layer, ctx_idx, feature), ...]
            - 'error_nodes': [(layer, ctx_idx), ...]
    """
    # First, find how many layers exist
    max_layer = max(attrs["layer"] for _, attrs in G.nodes(data=True))

    # Initialize structure
    per_layer_data = [
        {"feature_nodes": [], "error_nodes": []} for _ in range(max_layer + 1)
    ]

    # Fill in features and errors
    for node_id, attrs in G.nodes(data=True):
        feature_type = attrs.get("feature_type")
        layer = attrs.get("layer")
        ctx_idx = attrs.get("ctx_idx")

        if feature_type == "cross layer transcoder":
            feature = attrs.get("feature")
            per_layer_data[layer]["feature_nodes"].append((layer, ctx_idx, feature))
        elif feature_type == "mlp reconstruction error":
            per_layer_data[layer]["error_nodes"].append((layer, ctx_idx))

    return per_layer_data


def save_skeleton_json(skeleton_graph, initial_graph, file_path):
    """
    Save skeleton graph to JSON format, including internal connections within clusters
    and all node attributes from the original graph.
    
    Args:
        skeleton_graph: NetworkX graph from MapperPipeline (clusters as nodes)
        initial_graph: Original NetworkX graph with all node connections and attributes
        file_path: Path to save JSON file
    """
    nodes_data = []

    for cluster_id, data in skeleton_graph.nodes(data=True):
        cluster_nodes = data.get('nodes', [])

        # Prepare node data including all attributes
        full_nodes_data = []
        for node_id in cluster_nodes:
            if node_id in initial_graph.nodes:
                # Copy all attributes of the node
                node_attrs = dict(initial_graph.nodes[node_id])
                node_attrs["node_id"] = node_id  # ensure node_id is included
                full_nodes_data.append(node_attrs)
            else:
                # Fallback if node not found in initial graph
                full_nodes_data.append({"node_id": node_id})

        # Extract internal edges within the cluster
        internal_edges = []
        for u, v, edge_data in initial_graph.subgraph(cluster_nodes).edges(data=True):
            internal_edges.append({
                "source": u,
                "target": v,
                "weight": edge_data.get('weight', 1.0)
            })

        nodes_data.append({
            "cluster_id": str(cluster_id),
            "nodes": full_nodes_data,
            "size": len(full_nodes_data),
            "internal_edges": internal_edges
        })

    # Prepare skeleton links (between clusters)
    links_data = []
    for u, v, data in skeleton_graph.edges(data=True):
        links_data.append({
            "source": str(u),
            "target": str(v),
            "weight": data.get('weight', 1.0)
        })

    # Create final JSON structure
    output = {
        "nodes": nodes_data,
        "links": links_data
    }

    # Save to file
    with open(file_path, 'w') as f:
        json.dump(output, f, indent=2)


def save_graph_with_qparams(skeleton_graph, initial_graph, file_path):
    """
    Save graph in new format where the full initial graph is kept, and the skeleton
    contributes to 'qParams' with pinnedIds and supernodes.

    Args:
        skeleton_graph: NetworkX graph (clusters as nodes, with optional 'label' attribute)
        initial_graph: Original NetworkX graph with all node connections and attributes
        file_path: Path to save JSON file
    """

    # Collect pinnedIds = all nodes that belong to skeleton clusters
    pinned_ids = []
    supernodes = []

    for cluster_id, data in skeleton_graph.nodes(data=True):
        cluster_nodes = data.get("nodes", [])
        pinned_ids.extend(cluster_nodes)

        # Use cluster_id or "label" as the cluster label
        cluster_label = data.get("label", str(cluster_id))

        # Each supernode is a list: [label, node1, node2, ...]
        supernodes.append([cluster_label] + cluster_nodes)

    # Build qParams
    qparams = {
        "pinnedIds": pinned_ids,
        "supernodes": supernodes,
        "linkType": "both",
        "clickedId": "",
        "sg_pos": ""
    }

    # Build full node list from initial_graph
    nodes_data = []
    for node_id, attrs in initial_graph.nodes(data=True):
        node_entry = dict(attrs)
        node_entry["id"] = node_id
        nodes_data.append(node_entry)

    # Build full edge list from initial_graph
    links_data = []
    for u, v, attrs in initial_graph.edges(data=True):
        links_data.append({
            "source": u,
            "target": v,
            "weight": attrs.get("weight", 1.0)
        })

    # Final JSON structure
    output = {
        "qParams": qparams,
        "nodes": nodes_data,
        "links": links_data
    }

    # Save to file
    with open(file_path, "w") as f:
        json.dump(output, f, indent=2)

