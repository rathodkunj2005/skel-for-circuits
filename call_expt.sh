
# python -m experiments.run_experiment --prompt "Fact: The capital of the state containing Dallas is"
# python -m experiments.run_experiment --graph data/sample_graphs/gemma-fact-dallas-austin_2025-09-08T20-57-27-318Z.json


python -m experiments.run_experiment --graph data/sample_graphs/Haiku/sally-induction-qk.json --grouping llm;
python -m experiments.run_experiment --graph data/sample_graphs/Haiku/bomb-baseline.json;
python -m experiments.run_experiment --graph data/sample_graphs/Haiku/calc-6-plus-9.json;
python -m experiments.run_experiment --graph data/sample_graphs/Haiku/capital-state-oakland.json;
python -m experiments.run_experiment --graph data/sample_graphs/Haiku/cot-faithful-math-sqrt.json;
python -m experiments.run_experiment --graph data/sample_graphs/Haiku/medical-diagnosis-heart.json;
python -m experiments.run_experiment --graph data/sample_graphs/Haiku/karpathy-hallucination.json;
python -m experiments.run_experiment --graph data/sample_graphs/Haiku/michael-batkin-ha.json;
python -m experiments.run_experiment --graph data/sample_graphs/Haiku/bomb-comma.json;



