from tqdm import tqdm
import numpy as np
from mapper.pipeline import MapperPipeline
from .metrics import faithfulness_score, minimality_score
from data.loaders import extract_all_features_and_errors

from circuit_tracer import ReplacementModel, attribute
import torch

class Trainer:
    def __init__(self, graphs, lenses, n_intervals=10, overlap=0.2, lambda_param=0.5):
        self.graphs = graphs
        self.lenses = lenses
        self.n_intervals = n_intervals
        self.overlap = overlap
        self.lambda_param = lambda_param
        model_name = 'google/gemma-2-2b'
        transcoder_name = "gemma"
        self.model = ReplacementModel.from_pretrained(model_name, transcoder_name, device='cuda', dtype=torch.bfloat16)
        
    def objective(self, weights):
        """Evaluate weights on all training graphs."""
        total_faith = 0.0
        total_minimal = 0.0
        total_score = 0.0

        pipeline = MapperPipeline(
            self.lenses, 
            weights,
            self.n_intervals,
            self.overlap
        )
        for graph in self.graphs:
            skeleton = pipeline(graph)
        
            # faith = faithfulness_score(graph, skeleton)
            faith = faithfulness_score(self.model, extract_all_features_and_errors(skeleton))
            minimal = minimality_score(skeleton)

            total_faith += faith
            total_minimal += minimal
            total_score += faith + self.lambda_param * minimal

        
        n = len(self.graphs)
        return {
            "faithfulness": total_faith / n,
            "minimality": total_minimal / n,
            "score": total_score / n
        }
    
    def grid_search(self, weight_grid):
        # print(weight_grid)
        # print(". - "*10)
        # limit = 1
        
        best_score = -float('inf')
        best_weights = None
        best_metrics = None
        results = []
        
        
        for weights in tqdm(weight_grid):
            print("Trainer | Grid Search - weights:", weights)
            if len(weights) != len(self.lenses) or sum(weights) == 0:
                continue
            
            metrics = self.objective(weights) 
            score = metrics["score"]
            results.append((weights, metrics))
            
            if score > best_score:
                best_score = score
                best_weights = weights
                best_metrics = metrics
                
            print("Best weight combination so far:")
            print(best_weights)
            print("Metrics:")
            print(best_metrics)
            # limit -= 1
            # if limit == 0:
            #     break
                    
        return best_weights, best_metrics, results

