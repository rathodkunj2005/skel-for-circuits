import numpy as np
import json
import re
from transformers import pipeline
from data.loaders import get_output_nodes, get_top_logit_node


def remove_error_nodes(G):
    remove = []
    for node in G.nodes:
        if G.nodes[node]["feature_type"] == "mlp reconstruction error":
            remove.append(node)
    
    # print(f"Error nodes to be removed:\n{remove}")
    
    G_clean = G.copy()
    G_clean.remove_nodes_from(remove)
    return G_clean



    
def remove_extra_logit_nodes(G):
    output_nodes = get_output_nodes(G)
    
    keep = get_top_logit_node(G)
    
    print("-----> remove_extra_logit_nodes |", f"Keeping top logit node: {keep}, output_nodes: {output_nodes}")
    
    remove = []
    for o in output_nodes:
        if o != keep:
            remove.append(o)
    
    # print(f"Extra logit nodes to be removed:\n{remove}")
    G_clean = G.copy()
    G_clean.remove_nodes_from(remove)

    # print("remove_extra_logit_nodes |", G_clean)
    
    remaining_output_nodes = get_output_nodes(G_clean)
    print("-----> remove_extra_logit_nodes |", f"Remaining output nodes after extra_logit_nodes removal: {remaining_output_nodes}")
    
    return G_clean



def extract_token_from_node(node_data):
    """
    Extract clean token text from node data 'clerp' field.
    Handles 'Emb: "..."', '[...] Emb: "..."', etc.
    """
    raw_clerp = node_data.get("clerp", "")
    
    # Locate "Emb: "
    emb_marker = "Emb: "
    if emb_marker in raw_clerp:
        # Take everything after "Emb: "
        part = raw_clerp.split(emb_marker)[1]
        # It usually is quoted like "  make"
        # We want to remove the quotes if they exist
        if part.startswith('"') and part.endswith('"'):
            part = part[1:-1]
        return part.strip()
    return ""


def get_ordered_tokens(G):
    """
    Returns a list of (node_id, token_text) for all embedding nodes,
    ordered by context index.
    """
    embeddings = []
    for node in G.nodes():
        # Handle NetworkX 1.x vs 2.x compatibility
        if hasattr(G, "node"):
             data = G.node[node]
        else:
             data = G.nodes[node]
             
        if data.get("feature_type") == "embedding":
            # Extract ctx_idx
            # Usually in data['ctx_idx'], but sometimes encoded in ID "E_..._8"
            ctx_idx = data.get("ctx_idx")
            if ctx_idx is None:
                try:
                    parts = node.split('_')
                    ctx_idx = int(parts[-1])
                except:
                    ctx_idx = -1
            
            token_text = extract_token_from_node(data)
            embeddings.append((ctx_idx, node, token_text))
    
    # Sort by ctx_idx
    embeddings.sort(key=lambda x: x[0])
    return [(node, text) for _, node, text in embeddings]



def select_nodes_until_major_emb(G, lens, model="LiquidAI/LFM2.5-1.2B-Instruct", device="cuda", prompt_template=None, ignore_negative_scores=True):
    """
    Selects nodes based on lens scores (passed as 'lens' dict) until all 
    semantically important 'major' embedding nodes are included.
    
    Args:
        G: The graph.
        lens: A dictionary of {node_id: score}. User calls this 'lens values'.
        model: LLM model name. Qwen/Qwen3-32B, Qwen/Qwen3-4B-Instruct-2507, LiquidAI/LFM2.5-1.2B-Instruct, ...
        device: Device for LLM.
        ignore_negative_scores: If True, remove major nodes with negative scores from the list.
        
    Returns:
        int: The number of top nodes to keep (K) such that all major embedding nodes are in the top K.
    """
    # 1. Identify Major Tokens via LLM
    ordered_emb = get_ordered_tokens(G)
    if not ordered_emb:
        print("Warning: No embeddings found.")
        return 0
        
    full_text = " - ".join([text for _, text in ordered_emb])
    print(f"Reconstructed Text: {full_text}")
    
    if prompt_template is None:
        prompt = (
            f"Read the following text:\n\"{full_text}\"\n\n"
            "Task: Assume the goal is to guess the next word in the text.\n"
            "Select the most semantically important tokens that significantly contribute to predicting the next word.\n\n"
            "Constraints:\n"
            "- You MUST select no more than 35% of the input tokens.\n"
            "- Choose the most informative tokens only; avoid stop words and purely syntactic tokens unless absolutely necessary.\n"
            "- Think carefully and select the minimal-but-sufficient set under the 35% constraint.\n\n"
            "Output format:\n"
            "Format your response as a JSON object with exactly two fields:\n"
            "  - \"chosen_words\": a list of the selected tokens, in their original form\n"
            "  - \"explanation\": a concise explanation of why these tokens were chosen\n\n"
            "Example response format:\n"
            "{\n"
            "  \"chosen_words\": [\"word1\", \"word2\", \"word3\"],\n"
            "  \"explanation\": \"Brief explanation of why these tokens are semantically important.\"\n"
            "}\n\n"
            "Return ONLY valid JSON matching this structure.\n"
            "--> Response:"
        )
    else:
        prompt = prompt_template.format(text=full_text)
    
    try:
        generator = pipeline("text-generation", model=model, device=device)
        response = generator(prompt, max_new_tokens=2048, num_return_sequences=1, do_sample=False)
        output_text = response[0]['generated_text']
        
        # Cleanup: remove prompt if echoed
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):]
        
        print("***"*15, f"DEBUG: Output text:\n\n{output_text}\n\n")
        print("***"*15)

        # Parse JSON output
        json_start = output_text.find('{')
        # json_end = output_text.rfind('}') + 1 # this finds the last closing brace
        json_end = output_text.find('}') + 1 # this finds the first closing brace
        
        if json_start >= 0 and json_end > json_start:
            json_str = output_text[json_start:json_end]
            try:
                data = json.loads(json_str)
                major_token_strings = data.get("chosen_words", [])
                explanation = data.get("explanation", "")
                print(f"LLM Explanation: {explanation}")
                if isinstance(major_token_strings, str):
                    # In case it's a string representation of a list or single word
                    major_token_strings = [major_token_strings]
            except json.JSONDecodeError:
                print(f"Failed to parse JSON content: {json_str}")
                major_token_strings = []
        else:
            print(f"No JSON found in output: {output_text}")
            major_token_strings = []
            
        # Normalize to set of strings
        major_token_strings = set([str(s).strip() for s in major_token_strings if s])
        
    except Exception as e:
        print(f"Error calling LLM: {e}. Defaulting to all embeddings.")
        major_token_strings = set([t for _, t in ordered_emb if t])

    print(f"************* Major tokens identified: {major_token_strings}")

    # 2. Map major tokens to embedding nodes
    major_nodes = set()
    for node, text in ordered_emb:
        if text in major_token_strings:
            major_nodes.add(node)
            
    if ignore_negative_scores:
        filtered_major_nodes = set()
        scores = lens
        for node in major_nodes:
            score = scores.get(node, -float('inf'))
            if score >= 0:
                filtered_major_nodes.add(node)
            else:
                 print(f"Ignoring major node {node} due to negative score: {score}")
        major_nodes = filtered_major_nodes
            
    if not major_nodes:
        print("Warning: No major nodes matched (or all filtered). Returning 10% of nodes.")
        return int(len(G) * 0.1)

    print(f"************* Major nodes identified and found: {major_nodes}")

    # 3. Sort all nodes by score
    scores = lens
    # Ensure all nodes in graph have a score, default to -inf if missing
    sorted_nodes = sorted(G.nodes(), key=lambda n: scores.get(n, -float('inf')), reverse=True)
    
    # 4. Find cutoff
    max_rank = 0
    found_count = 0
    total_major = len(major_nodes)
    
    all_embedding_nodes = set([node for node, _ in ordered_emb])

    for i, node in enumerate(sorted_nodes):
        if node in all_embedding_nodes:
            is_major = " (MAJOR)" if node in major_nodes else ""
            
            # Helper to get node data safely
            if hasattr(G, "node"):
                node_data = G.node[node]
            else:
                node_data = G.nodes[node]
            
            clerp_val = node_data.get("clerp", "")
            print(f"--> Found embedding node {node} with score {scores.get(node)}{is_major} | clerp: {clerp_val}")
            
        if node in major_nodes:
            max_rank = i
            found_count += 1
            if found_count == total_major:
                break
    
    print(f"Found {found_count}/{total_major} major nodes. Cutoff at rank {max_rank} (Top {max_rank + 1} nodes) out of {len(sorted_nodes)} nodes.")
    
    # Return count to keep
    return max_rank + 1

       

def prune_graph_by_one_lens(G, lens):
    """Remove error nodes and nodes with non-positive lens score."""
    G = remove_error_nodes(G)
    scores = {node: lens(G, node) for node in G.nodes()}
    nodes_to_remove = [node for node in G.nodes() if scores[node] <= 0]
    G.remove_nodes_from(nodes_to_remove)
    return G


def create_cover(points, n_intervals=10, overlap=0.2):
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [(points[0], points[0]+1)]
    if np.max(points) - np.min(points) < 1e-8: # all points are the same
        return [(points[0], points[0]+1)]
    
    min_val = np.min(points)
    max_val1 = np.max(points)
    points_without_max = points[points != max_val1]
    max_val = np.max(points_without_max)
    
    range_val = max_val - min_val
    interval_length = range_val / (n_intervals - overlap * (n_intervals - 1))
    step = interval_length * (1 - overlap) - 0.1
    
    intervals = []
    start = min_val
    for _ in range(n_intervals):
        end = start + interval_length
        intervals.append((start, end))
        start = start + step
    
    # update last interval
    intervals[-1] = (intervals[-1][0], max_val1+1)
    
    return intervals
