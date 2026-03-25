import requests
import json
import os
import networkx as nx
from data.supernode_label import get_umbrella_term

output_nodes_global = []
 

def load_graph(graph):
    """Convert graph to NetworkX DiGraph."""
    if isinstance(graph, str):
        with open(graph, 'r') as f:
            data = json.load(f)
    else:
        data = graph
    
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
                # influence=node['influence'],
                # activation=node['activation']
            )
            
    # Add edges
    for link in data['links']:
        G.add_edge(link['source'], link['target'], weight=link['weight'])
    
    print(f"---> Loaded the input graph with {len(G.nodes())} nodes and {len(G.edges())} edges.")

    
    # Store graph-level metadata so downstream code (e.g. rerun_auto_interpretation)
    # can look up feature data using the correct model/scan identifier.
    metadata = data.get("metadata", {})
    G.graph["scan"] = metadata.get("scan", "")
    G.graph["neuronpedia_source_set"] = (
        metadata.get("info", {}).get("neuronpedia_source_set") or ""
    )

    for node in G.nodes():
        feature_type = G.nodes[node]["feature_type"]
        if "logit" in feature_type.lower():
            print("LOGIT NODE:", node, G.nodes[node]['token_prob'])
    t = get_top_logit_node(G)
    print("TOP LOGIT NODE:", t, G.nodes[t]['token_prob'])

    return G


def load_all_graphs(directory):
    """Load all JSON graphs from a directory."""
    graphs = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            graphs.append(load_graph(filepath))
    return graphs


def pin_node_to_input_ratio_manual_graph(graph_file):
    # use median of vignette to graph node ration
    # Gemma & Qwen: 0.0265
    # Haiku: 0.301
    if "haiku" in graph_file.lower():
        ratio = 0.301
    elif "clt" in graph_file.lower():
        ratio = 0.16375
    else:
        ratio = 0.0265
    
    with open(graph_file, 'r') as f:
        data = json.load(f)
        pin = len(data['qParams']['pinnedIds'])
        node_count = len(data['nodes'])
        return pin, round(node_count*ratio)
    
    # Older version: get vignette to input token ratio
    # with open(graph_file, 'r') as f:
    #     data = json.load(f)
    # try:
    #     pin = len(data['qParams']['pinnedIds'])
    # except:
    #     pin = 0
        
    # try:
    #     input_node = len(data['metadata']['prompt_tokens'])
    # except:
    #     input_node = 0
        
    # try:
    #     return round(pin/input_node, 10), pin
    # except:
    #     return 2, pin
    
    
def get_input_nodes(G):
    """Identify input nodes based on node_id pattern."""
    input_nodes = []
    for node in G.nodes():
        feature_type = G.nodes[node]["feature_type"]
        if "embedding" in feature_type.lower():
            input_nodes.append(node)
    return input_nodes
    

def get_output_nodes(G):
    output_nodes = []
    for node in G.nodes():
        feature_type = G.nodes[node]["feature_type"]
        if "logit" in feature_type.lower():
            output_nodes.append(node)
            
    return output_nodes


def get_top_logit_node(G):
    output_nodes = get_output_nodes(G)
    print("get_top_logit_node | output_nodes:", output_nodes)
    keep = ''
    prob = 0.0
    
    for ot in output_nodes:
        try:
            if G.nodes[ot]['token_prob'] > prob:
                prob = G.nodes[ot]['token_prob']
                keep = ot
        except:
            pass
    
    # print("\t\t\t\t\t\t\t\t\t\t\ttop_logit_node ~~~> ", keep)  
    return keep


def get_layer_wise_nodes(G):
    layer_wise_nodes = {}
    for node in G.nodes():
        try:
            prefix = int(node.split('_')[0])
        except:
            prefix = -1
            
        layer_wise_nodes[prefix] = layer_wise_nodes.get(prefix, [])
        layer_wise_nodes[prefix].append(node)
    
    layer_wise_nodes = dict(sorted(layer_wise_nodes.items()))
    return layer_wise_nodes
    

def get_incoming_nodes_with_weights(G, target_node):
    """Get nodes with incoming edges to target node along with edge weights."""
    incoming_data = []
    for predecessor in G.predecessors(target_node):
        weight = G[predecessor][target_node]['weight']
        incoming_data.append({
            'node_id': predecessor,
            'weight': weight
        })
    return incoming_data


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

        if feature_type == "mlp reconstruction error":
            selected_errors.append((layer, ctx_idx))
        else: # feature_type == "cross layer transcoder":
            feature = attrs.get("feature")
            selected_features.append((layer, ctx_idx, feature))
        

    g_dict = {
        "feature_nodes": selected_features,
        "error_nodes": selected_errors
    }
    
    # print(len(selected_features), len(selected_errors))
    return g_dict


def save_graph_with_qparams(skeleton_graph, initial_graph, metadata, file_path, use_closed_source_labeling: bool = True):
    """
    Save graph in new format where the full initial graph is kept, and the skeleton
    contributes to 'qParams' with pinnedIds and supernodes.

    Args:
        skeleton_graph: NetworkX graph (clusters as nodes, with optional 'label' attribute)
        initial_graph: Original NetworkX graph with all node connections and attributes
        file_path: Path to save JSON file
        use_closed_source_labeling: If True, use the closed-source (OpenAI) model for
            get_umbrella_term; otherwise use the local open-source model.
    """

    # Collect pinnedIds = all nodes that belong to skeleton clusters
    pinned_ids = []
    supernodes = []

    print("--> DATA LAODER/SENDER: Skeleton graph id:", id(skeleton_graph), "Number of clusters:", skeleton_graph.number_of_nodes(), "Initial graph id:", id(initial_graph), "Number of nodes in initial graph:", initial_graph.number_of_nodes())
    for cluster_id, data in skeleton_graph.nodes(data=True):
        cluster_nodes = data.get("nodes", [])
        pinned_ids.extend(cluster_nodes)

        # Use cluster_id or "label" as the cluster label
        cluster_label = data.get("label", str(cluster_id))
        
        # Each supernode is a list: [ label, node1, node2, ..., noden]
        # supernodes.append([cluster_label] + cluster_nodes)
        if len(cluster_nodes) > 1:
            # If the skeleton node already carries a descriptive label (e.g. from
            # LLMGroupingPipeline), use it directly and skip the extra LLM call.
            # Otherwise fall back to the standard get_umbrella_term path.
            pre_set = data.get("label")
            if pre_set and pre_set != str(cluster_id) and pre_set != "[unlabeled]":
                umbrella = pre_set
            else:
                # all_labels = [initial_graph.nodes[x]["clerp"] for x in cluster_nodes]
                all_labels = [initial_graph.nodes[x]["label"] for x in cluster_nodes]
                # umbrella = get_umbrella_term(all_labels, model="Qwen/Qwen2.5-32B-Instruct")
                umbrella = get_umbrella_term(all_labels, use_closed_source=use_closed_source_labeling)
            supernodes.append([umbrella] + cluster_nodes)
        elif cluster_nodes[0] != '':
            # supernodes.append([initial_graph.nodes[cluster_nodes[0]]["clerp"]] + cluster_nodes)
            supernodes.append([initial_graph.nodes[cluster_nodes[0]]["label"]] + cluster_nodes)
        else:
            print("Empty cluster:", cluster_nodes)
            
    # Build qParams
    qparams = {
        "linkType": "both",
        "pinnedIds": pinned_ids,
        "clickedId": "",
        "supernodes": supernodes,
        "sg_pos": ""
    }

    # Build full node list from initial_graph
    nodes_data = []
    for node_id, attrs in initial_graph.nodes(data=True):
        node_entry = dict(attrs)
        node_entry["node_id"] = node_id
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
        "metadata": metadata,
        "qParams": qparams,
        "nodes": nodes_data,
        "links": links_data
    }

    # Save to file
    with open(file_path, "w") as f:
        json.dump(output, f, indent=4)
    
    return output


def send_subgraph_to_api(graph, displayName=""):
    

    url = "https://www.neuronpedia.org/api/graph/subgraph/save"
    # my_key = "sk-np-05sJukY6531bHuEinDGx9K6hqEftfQsWIaTL0BoNnWM0"
    my_key = "sk-np-PbKa0gIJZ0DMePXpxS7pso5PRexTfTZtNdC0OduMbvg0"
    
    slug = graph["metadata"]["slug"]
    
    pinnedIds = graph["qParams"]["pinnedIds"]
    supernodes = graph["qParams"]["supernodes"]
    pruningThreshold = 1
    densityThreshold = 1
    
    
    # print("# # # # pinnedIds")
    # print(pinnedIds)
    
    modelId = graph["metadata"]["scan"]
    node_lookup = {node.get("node_id"): node for node in graph.get("nodes", [])}
    
    if "gemma" not in modelId and "qwen" not in modelId: # for Haiku
        # # # Temp Fix for Haiku
        new_pinnedIds = []
        for node in pinnedIds:
            new_pinnedIds.append(replace_node_id_with_JSid(node, node_lookup))
        
        new_supernodes = []
        for group in supernodes:
            s = [group[0]]
            for node in group[1:]:
                s.append(replace_node_id_with_JSid(node, node_lookup))
            new_supernodes.append(s)  
        
        pinnedIds = new_pinnedIds
        supernodes = new_supernodes
        # # # 
    
    if displayName == "":
        displayName = slug + "-" + str(len(pinnedIds)) + "-1D-imp"
        
    json_file = {
        "modelId": modelId,
        "slug": slug,
        "displayName": displayName,
        "pinnedIds": pinnedIds,
        "supernodes": supernodes,
        # "pinnedIds": new_pinnedIds,
        # "supernodes": new_supernodes,
        "clerps": [],
        "pruningThreshold": pruningThreshold,
        "densityThreshold": densityThreshold,
        "overwriteId": ""
    }

    # print("++++++++++++++++")
    # print(json.dumps(json_file, indent=2))
    # print("++++++++++++++++")
    
    x = requests.post(
        url,
        headers={
        "Content-Type": "application/json",
        "x-api-key": my_key
        },
        json=json_file
    )

    print("\t---> Tried sending a subgraph:", slug + "-" + str(len(pinnedIds)), "to attr. graph:", slug, "display name:", displayName)
    print(x)


def replace_node_id_with_JSid(node_id: str, node_lookup: dict[str, dict]) -> str:
        """Return the frontend jsNodeId for node_id when available."""
        node_attrs = node_lookup.get(node_id)
        if isinstance(node_attrs, dict):
            js_id = node_attrs.get("jsNodeId") or node_attrs.get("jsnodeid")
            if js_id:
                return js_id
        return node_id


def stat_check(d):
    for filename in os.listdir(d):
        if filename.endswith(".json"):
            with open(d+filename) as f:
                data = json.load(f)
                orig_pinned_nodes = set(list(data['qParams']['pinnedIds']))
                # print(f"{filename}\nNode count: {len(data['nodes'])}\t prompt: {len(data['metadata']['prompt_tokens'])}\t circuit: {len(data['qParams']['pinnedIds'])}")
                supernodes = data['qParams']['supernodes']
                supernodes_stat = []
                for s in supernodes:
                    if len(s) > 1:
                        supernodes_stat.append(len(s))
                if len(supernodes_stat) != 0:
                    print(f"Avg node count in supernodes with >1 node: {sum(supernodes_stat)/len(supernodes_stat)}")
                    # print(len(supernodes_stat))
            continue
        
            try:
                out_d = d.replace("sample_graphs", "outputs")
                filename = filename.replace(".json", "_skeleton.json")
                print(out_d+filename, end=", ")
                with open(out_d+filename) as f:
                    data = json.load(f)
                    auto_pinned_nodes = set(list(data['qParams']['pinnedIds']))
                    node_lookup = {node.get('node_id'): node for node in data.get('nodes', [])}
                    # print(f"{filename}\nNode count: {len(data['nodes'])}\t prompt: {len(data['metadata']['prompt_tokens'])}\t circuit: {len(data['qParams']['pinnedIds'])}")
                
                if "Haiku" in out_d:
                    auto_pinned_nodes_fixed = []
                    for n in auto_pinned_nodes:
                        auto_pinned_nodes_fixed.append(replace_node_id_with_JSid(n, node_lookup))
                
                auto_pinned_nodes_fixed = set(auto_pinned_nodes_fixed)

                common_nodes = set.intersection(orig_pinned_nodes, auto_pinned_nodes_fixed)
                print(f"initial: {len(orig_pinned_nodes)}, auto: {len(auto_pinned_nodes)}, intersect: {len(common_nodes)}")
            except:
                pass
            

# print("~~~~~"*20)
# print("GRAPH STAT CHECK")
# d = "../data/sample_graphs/Haiku/"
# stat_check(d)
# print("~~~~~"*20)
# d = "../data/sample_graphs/Gemma/"
# stat_check(d)
# print("~~~~~"*20)
# d = "data/sample_graphs/"
# stat_check(d)
# print("GRAPH END OF STAT CHECK")
# print("~~~~~"*20)
# print()
# exit()
