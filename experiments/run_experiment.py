import os
import json
import numpy as np
import math
import argparse
from data.loaders import load_graph, load_all_graphs, save_skeleton_json, save_graph_with_qparams
from data.loaders import get_first_predicted_token
from apply.apply_pipeline import apply_trained_weights
from lenses.depth import DepthLens
from lenses.supernode import SupernodeLens
from lenses.importance import ImportanceLens
from training.trainer import Trainer
from mapper.utils import prune_graph_by_lens_combination, prune_graph_by_one_lens

from circuit_tracer import ReplacementModel, attribute

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

GRID_STEPS = 5
APPLICATION_SEARCH_SIZE = 1

def build_weights(start, end):
    # Generate weight grid (normalized to sum to 1)
    weights = np.linspace(start, end, GRID_STEPS)
    weight_grid = []
    
    for w1 in weights:
        for w2 in weights:
            for w3 in weights:
                total = w1 + w2 + w3
                if total > 0:  # Avoid division by zero
                    normalized = [w1/total, w2/total, w3/total]
                    if normalized not in weight_grid:
                        weight_grid.append(normalized)
    return weight_grid


def refine_weights(b1, b2, b3, max_combinations, r=0.05):
    """Refined search around best weights (b1, b2, b3)"""
    # Adaptive radius based on distance from boundaries
    radius = r/2 if min(b1, b2, b3) < r or max(b1, b2, b3) > 1-r else r

    steps = int(max_combinations ** (1/3)) + 1
    
    # Generate grid around best point
    ranges = [np.linspace(max(0, b - radius), min(1, b + radius), steps) 
              for b in [b1, b2, b3]]
    
    # Create normalized combinations
    combinations = []
    for w1 in ranges[0]:
        for w2 in ranges[1]:
            for w3 in ranges[2]:
                total = w1 + w2 + w3
                if total > 0:
                    combinations.append([w1/total, w2/total, w3/total])
    
    # Remove duplicates and limit to max_combinations
    unique = []
    for combo in combinations:
        if not any(all(abs(a-b) < 1e-6 for a, b in zip(combo, existing)) 
                  for existing in unique):
            unique.append(combo)
            if len(unique) >= max_combinations:
                break
    
    return unique


def main():

    parser = argparse.ArgumentParser(
        description="Apply trained mapper skeletonization to a graph"
    )

    parser.add_argument("--prompt", required=True, type=str, 
                        help="input prompt")    
    parser.add_argument("--desired_logit_prob", type=float, default=0.95,
                        help="")
    parser.add_argument("--max_n_logits", type=int, default=10,
                        help="")    
    parser.add_argument("--max_feature_nodes", type=int, default=8192,
                        help="")    
    parser.add_argument("--graph_batch_size", type=int, default=256,
                        help="")    
                        
    parser.add_argument("--do_pretrain", type=bool, default=False,
                        help="Train the weights for combining lenses")
    parser.add_argument("--graph", required=True,
                        help="Path to input graph JSON")
    parser.add_argument("--train_data_dir", type=str, 
                        help="Path to graph JSONs for training the weights")    
    parser.add_argument("--num_intervals", type=int, default=8,
                        help="Number of intervals in cover")
    parser.add_argument("--overlap", type=float, default=0.2,
                        help="Overlap fraction in cover")
    parser.add_argument("--lambda_param", type=float, default=0.5,
                        help="Combining faithfulness and minimality")
    parser.add_argument("--prune_fraction", type=float, default=0.5,
                        help="Fraction of nodes with lowest importance scores to be remove from the graph")

                        
    # parser.add_argument("--weights", required=True,
    #                     help="Path to JSON file with lens weights")
    # parser.add_argument("--supernode_threshold", type=float, default=0.7,
    #                     help="Threshold similarity of node labels to be grouped into supernodes.")
    # parser.add_argument("--cluster_method", choices=["l2","agglo"], default="l2",
    #                     help="Clustering method for mapper")
    # parser.add_argument("--out", required=True,
    #                     help="Path to save skeleton JSON")

    args = parser.parse_args()

    if args.do_pretrain and not args.train_data_dir:
        parser.error("--train_data_dir is required when --do_train is specified.")
    
    # Initialize lenses
    lenses = [
        DepthLens(),
        SupernodeLens(),
        ImportanceLens()
    ]


    model_name = 'google/gemma-2-2b'
    transcoder_name = "gemma"
    model = ReplacementModel.from_pretrained(model_name, transcoder_name, device='cuda', dtype=torch.bfloat16)
    


    # ----------------------------
    # --- PHASE 1: PRE-TRAINING ---
    if args.do_pretrain:
        print("---> Start Training...")
        # Load training graphs
        print("Loading training graphs...")
        train_graphs = load_all_graphs(args.train_data_dir)
        
        # print(f"Pruning: Removing {args.prune_fraction*100}% of nodes with lowest importance scores to be remove from the graph...")
        # pruned_train_graphs = []
        # for G in train_graphs:
        #     pruned_train_graphs.append(prune_graph_by_one_lens(G, lenses[-1], args.prune_fraction))

        # print(f"Loaded {len(pruned_train_graphs)} training graphs")


        # Train weights
        # trainer = Trainer(pruned_train_graphs, lenses, args.num_intervals, args.overlap, args.lambda_param)
        trainer = Trainer(train_graphs, lenses, args.num_intervals, args.overlap, args.lambda_param)
        
        weight_grid = build_weights(0, 1)

        print(f"Generated {len(weight_grid)} weight combinations")
        
        # Perform grid search
        print("Starting grid search...")

        # best_weights, best_score = [0.2, 0.4, 0.4], 0.912

        best_weights, best_metrics, all_results = trainer.grid_search(weight_grid)

        # Save results
        results = {
            "best_weights": {
                "DepthLens": best_weights[0],
                "SupernodeLens": best_weights[1],
                "ImportanceLens": best_weights[2]
            },
            "best_metrics": best_metrics,
            "all_results": [
                {"weights": w, "metrics": m} 
                for w, m in all_results
            ],
            "config": {
                "lambda_param": args.lambda_param,
                "grid_steps": GRID_STEPS,
                "num_graphs": len(train_graphs)
            }
        }
        
        output_path = os.path.join("./training", "training_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved results to {output_path}")
        print(f"Best weights: {best_weights}")
        print(f"Faithfulness: {best_metrics['faithfulness']:.4f}")
        print(f"Minimality: {best_metrics['minimality']:.4f}")
        print(f"Composite Score: {best_metrics['score']:.4f}")


    # ----------------------------
    # --- PHASE 2: APPLICATION ---
    print("---> Skeletonizing...")
                    
    # Load test graphs
    print("Loading test graphs...")
    test_graphs = [load_graph(args.graph)]
    print(f"Loaded {len(test_graphs)} test graphs")

    # print(f"Pruning: Removing {args.prune_fraction*100}% of nodes with lowest importance scores to be remove from the graph...")
    # pruned_test_graphs = []
    # for G in test_graphs:
    #     pruned_test_graphs.append(prune_graph_by_importance(G, args.prune_fraction))
        
    weights_path = "training/training_results.json"
    # Load best weights from training
    with open(weights_path, 'r') as f:
        weights_data = json.load(f)
        best_weights = weights_data["best_weights"]


    refined_weights_list = refine_weights(best_weights[0], best_weights[1], best_weights[2], max_combinations=APPLICATION_SEARCH_SIZE, r=0.1)
    print(refined_weights_list)

    if APPLICATION_SEARCH_SIZE > 0:
        # Perform grid search
        print("Starting grid search...")
        trainer = Trainer(test_graphs, lenses, args.num_intervals, args.overlap, args.lambda_param)
        best_weights, best_metrics, all_results = trainer.grid_search(refined_weights_list)
        # best_weights, best_score = [0.2, 0.4, 0.4], 0.912

    # Apply to each test graph
    print("Generating skeletons...")
    for i, graph in enumerate(test_graphs):
        print(f"Processing graph {i+1}/{len(test_graphs)}")
        
        # Apply mapper pipeline with trained weights
        skeleton = apply_trained_weights(
            graph=graph,
            lenses=lenses,
            trained_weights=best_weights,
            n_intervals=10,
            overlap=0.2
        )
        
        # Save skeleton
        base_name = os.path.splitext(os.path.basename(args.graph))[0]
        output_path = os.path.join("data/outputs/", f"{base_name}_skeleton.json")
        save_skeleton_json(skeleton, graph, output_path)
        save_graph_with_qparams(skeleton, graph, output_path)
        print(f"Saved skeleton to {output_path}")
    
    print("\n=== PROCESS COMPLETE ===")
    print(f"Generated {len(test_graphs)} skeletons in data/outputs/")


if __name__ == "__main__":
    main()


## Call Seq.
# create lenses                                                     lenses/
# load graphs                                                       data/loaders.py
# create trainer                                                    training/trainer.py
#   trainer grid search
#       trainer objective
#           create MapperPipeline
#           call it
#               compute lenses
#               get intervals
#               create new skeleton
#           get faith an minimal score based on the skeleton
#           return combined score
#       decide on the current weights combination
#       return best weights combination
