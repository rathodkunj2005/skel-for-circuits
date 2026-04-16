import argparse
import csv
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


STATES = [
    ("Alabama", "Montgomery", "Birmingham"),
    ("Alaska", "Juneau", "Anchorage"),
    ("Arizona", "Phoenix", "Tucson"),
    ("California", "Sacramento", "Los Angeles"),
    ("Florida", "Tallahassee", "Miami"),
    ("Illinois", "Springfield", "Chicago"),
    ("Kansas", "Topeka", "Wichita"),
    ("Kentucky", "Frankfort", "Louisville"),
    ("Louisiana", "Baton Rouge", "New Orleans"),
    ("Michigan", "Lansing", "Detroit"),
    ("Minnesota", "Saint Paul", "Minneapolis"),
    ("Montana", "Helena", "Billings"),
    ("Nevada", "Carson City", "Las Vegas"),
    ("New Mexico", "Santa Fe", "Albuquerque"),
    ("New York", "Albany", "New York City"),
    ("North Carolina", "Raleigh", "Charlotte"),
    ("Ohio", "Columbus", "Cincinnati"),
    ("Oregon", "Salem", "Portland"),
    ("Pennsylvania", "Harrisburg", "Philadelphia"),
    ("Texas", "Austin", "Dallas"),
    ("Washington", "Olympia", "Seattle"),
    ("Wisconsin", "Madison", "Milwaukee"),
]

DEFAULT_MODEL = "google/gemma-2-2b"


@dataclass(frozen=True)
class EvaluationResult:
    prompt: str
    correct_answer: str
    p_correct: float
    margin: float
    correct_rank: int
    best_incorrect_answer: str
    p_best_incorrect: float
    top_first_token: str
    p_top_first_token: float
    top_completion: str


def create_alternate_world() -> list[tuple[tuple[str, str], str]]:
    states = [(state, alternate_city) for state, _, alternate_city in STATES]
    random.shuffle(states)

    capitals = [capital for _, capital, _ in STATES]
    random.shuffle(capitals)

    return list(zip(states, capitals))


def create_alternate_world_prompt(world: list[tuple[tuple[str, str], str]]) -> str:
    prompt = "In this world,"
    for (state, _), capital in world[:-1]:
        prompt += f" {state}'s capital is {capital},"
    prompt += f" and {world[-1][0][0]}'s capital is {world[-1][1]}.\n"
    return prompt


def create_two_hop_reasoning_prompt(
    world: list[tuple[tuple[str, str], str]],
) -> tuple[str, str]:
    prompt = create_alternate_world_prompt(world)
    (_, alternate_city), capital = random.choice(world)
    prompt += f"The capital of the state containing {alternate_city} is"
    return prompt, capital


def _answer_variants(answer: str) -> list[str]:
    base = answer.strip()
    title_case = base.capitalize()
    return [base, f" {base}", title_case, f" {title_case}"]


def _load_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
    model.eval()
    return tokenizer, model


def _sequence_logprob(prompt: str, completion: str, tokenizer, model) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    full_ids = tokenizer(
        prompt + completion, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    prompt_len = prompt_ids.shape[-1]
    completion_len = full_ids.shape[-1] - prompt_len
    if completion_len <= 0:
        raise ValueError(
            f"Completion must add tokens. Prompt={prompt!r}, completion={completion!r}"
        )

    full_ids = full_ids.to(model.device)
    with torch.no_grad():
        logits = model(full_ids).logits[:, :-1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)

    target_ids = full_ids[:, 1:]
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(
        -1
    )
    completion_log_probs = token_log_probs[:, prompt_len - 1 :]
    return float(completion_log_probs.sum().item())


def _top_first_token(prompt: str, tokenizer, model) -> tuple[str, float]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(model.device)
    with torch.no_grad():
        next_token_logits = model(prompt_ids).logits[:, -1, :].float()
        next_token_probs = torch.softmax(next_token_logits, dim=-1)

    top_token_id = int(next_token_probs.argmax(dim=-1).item())
    top_token_prob = float(next_token_probs[0, top_token_id].item())
    top_token = tokenizer.decode([top_token_id])
    return top_token, top_token_prob


def _top_completion(prompt: str, tokenizer, model, max_new_tokens: int = 3) -> str:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(model.device)
    generated_ids = prompt_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            next_token_logits = model(generated_ids).logits[:, -1, :].float()
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            if token_text.endswith((".", ",", "\n", ":")):
                break

    completion_ids = generated_ids[0, prompt_ids.shape[-1] :].tolist()
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion


def _aggregate_variant_scores(prompt: str, answer: str, tokenizer, model) -> float:
    variant_scores = [
        _sequence_logprob(prompt, variant, tokenizer, model)
        for variant in _answer_variants(answer)
    ]
    return float(
        torch.logsumexp(torch.tensor(variant_scores, dtype=torch.float64), dim=0).item()
    )


def _score_candidates(prompt: str, tokenizer, model) -> dict[str, float]:
    return {
        capital: _aggregate_variant_scores(prompt, capital, tokenizer, model)
        for _, capital, _ in STATES
    }


def evaluate_prompt(
    prompt: str, correct_answer: str, tokenizer, model
) -> EvaluationResult:
    candidate_scores = _score_candidates(prompt, tokenizer, model)
    top_first_token, top_first_token_prob = _top_first_token(prompt, tokenizer, model)
    top_completion = _top_completion(prompt, tokenizer, model)

    scores_tensor = torch.tensor(list(candidate_scores.values()), dtype=torch.float64)
    log_normalizer = torch.logsumexp(scores_tensor, dim=0).item()
    probabilities = {
        answer: math.exp(score - log_normalizer)
        for answer, score in candidate_scores.items()
    }

    sorted_answers = sorted(
        candidate_scores.keys(),
        key=lambda currency: candidate_scores[currency],
        reverse=True,
    )
    best_incorrect_answer = next(
        answer for answer in sorted_answers if answer != correct_answer
    )
    p_correct = probabilities[correct_answer]
    p_best_incorrect = probabilities[best_incorrect_answer]

    return EvaluationResult(
        prompt=prompt,
        correct_answer=correct_answer,
        p_correct=p_correct,
        margin=p_correct - p_best_incorrect,
        correct_rank=sorted_answers.index(correct_answer) + 1,
        best_incorrect_answer=best_incorrect_answer,
        p_best_incorrect=p_best_incorrect,
        top_first_token=top_first_token,
        p_top_first_token=top_first_token_prob,
        top_completion=top_completion,
    )


def _write_results_csv(results: list[EvaluationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys())
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate alternate-world two-hop reasoning prompts with candidate-normalized answer probabilities."
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of random prompts to evaluate.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/evaluation_results/two_hop_reasoning_results.csv"),
        help="CSV path for evaluation results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to evaluate (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    args = parser.parse_args()
    if args.num_trials < 1:
        parser.error("--num_trials must be at least 1")
    return args


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    tokenizer, model = _load_model_and_tokenizer(args.model)

    results = []
    for _ in range(args.num_trials):
        world = create_alternate_world()
        prompt, correct_answer = create_two_hop_reasoning_prompt(world)
        result = evaluate_prompt(prompt, correct_answer, tokenizer, model)
        results.append(result)
        print(
            f"answer={result.correct_answer} p_correct={result.p_correct:.4f} "
            f"margin={result.margin:.4f} rank={result.correct_rank}"
        )

    _write_results_csv(results, args.output_csv)
    print(f"Wrote {len(results)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
