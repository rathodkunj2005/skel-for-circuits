import json
import re
import requests

# ---------------------------------------------------------------------------
# Model configuration
# Pass use_closed_source=True to rerun_auto_interpretation / get_umbrella_term
# (or directly to generate_text) to route generation through the OpenAI API.
# ---------------------------------------------------------------------------

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OpenAI model used when use_closed_source=True
CLOSED_SOURCE_MODEL = "gpt-5.2-pro-2025-12-11"  
CLOSED_SOURCE_MODEL = "gpt-5.2-pro"  

OPEN_SOURCE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# OPEN_SOURCE_MODEL = "Qwen/Qwen3-32B"
# OPEN_SOURCE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
# OPEN_SOURCE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Lazily-initialized singletons (created on first use).
_openai_client = None
_hf_generator = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"supernode labeling: Initialized OpenAI client (model='{CLOSED_SOURCE_MODEL}').")
    return _openai_client


def _get_hf_generator():
    global _hf_generator
    if _hf_generator is None:
        from transformers import pipeline
        _hf_generator = pipeline("text-generation", model=OPEN_SOURCE_MODEL)
        print(f"supernode labeling: Loaded HuggingFace model '{OPEN_SOURCE_MODEL}'.")
    return _hf_generator


def generate_text(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    use_closed_source: bool = True,
) -> str:
    """
    Unified text-generation helper.

    Args:
        prompt: The input prompt string.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        use_closed_source: If True, use the OpenAI API (CLOSED_SOURCE_MODEL).
                           If False, use the local HuggingFace pipeline (OPEN_SOURCE_MODEL).

    Returns the generated text string (prompt excluded for HF models).
    """
    if use_closed_source:
        client = _get_openai_client()
        response = client.responses.create(
            model=CLOSED_SOURCE_MODEL,
            input=prompt,
            reasoning={
                "effort": "high" # "none", "low", "medium", or "high", indicating how much reasoning the model needs to do
            },
            text={
                "verbosity": "high" # "low", "medium", or "high",  how many output tokens are generated. Lowering the number of tokens reduces overall latency.
            },
            # temperature=temperature, # only supported when using GPT-5.2 with reasoning effort set to none:
        )
        # print(f"supernode labeling: Received response from OpenAI model '{CLOSED_SOURCE_MODEL}'.")
        # print(f"--DEBUG: Full response object:\n{response}\n")
        # print(f"--DEBUG: Raw generated text:\n{response.output_text.strip()}\n")
        return response.output_text.strip()
    else:
        max_tokens = max_tokens * 2 # No cost, so generate more to increase chance of getting a valid JSON response
        generator = _get_hf_generator()
        response = generator(
            prompt,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=1.0,
        )
        # HF pipelines return the full text including the prompt; strip it.
        full_text: str = response[0]["generated_text"]
        return full_text[len(prompt):]


def parse_json_response(raw: str, key: str) -> str | None:
    """
    Robustly extract `key` from a JSON object in `raw`.

    Strategy:
    1. Strip <think>…</think> reasoning blocks (open-source models often emit these).
    2. Try to find a JSON object — first inside a ```json … ``` fence, then bare.
    3. Parse with json.loads and return the value for `key`.
    4. As a last resort, try a simple regex to pull the value directly.

    Returns the string value, or None if extraction fails.
    """
    # 1. Remove <think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # 2a. Try fenced code block first
    fence_match = re.search(r"```(?:json)?\s*({.*?})\s*```", cleaned, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else None

    # 2b. Find the last {...} block (models sometimes emit preamble before JSON)
    if candidate is None:
        brace_matches = list(re.finditer(r"\{[^{}]*\}", cleaned, re.DOTALL))
        if brace_matches:
            candidate = brace_matches[-1].group(0)

    # 3. Parse JSON
    if candidate:
        try:
            obj = json.loads(candidate)
            value = obj.get(key, "").strip()
            if value:
                return value
        except json.JSONDecodeError:
            pass

    # 4. Regex fallback: "key": "value" or "key": value
    pattern = rf'["|\']?{re.escape(key)}["|\']?\s*:\s*["\']([^"\'\n]+)["\']'
    regex_match = re.search(pattern, cleaned, re.IGNORECASE)
    if regex_match:
        return regex_match.group(1).strip()

    return None


def format_activating_example(tokens, activations):
    """
    Format a single activating example for use in a prompt.
    Tokens with activation > 0 are wrapped in <<>> and a second line lists
    only the non-zero (token, activation%) pairs.

    Returns a formatted string like:
        Text:  and he was <<over the moon>> to find
        Activations: ('over', 5), (' the', 6), (' moon', 9)

    Returns None if no token has activation > 0 (example is uninformative).
    """
    assert len(tokens) == len(activations), "Lists must have same length."

    if not any(a > 0 for a in activations):
        return None

    # Build highlighted text
    highlighted = []
    i = 0
    n = len(tokens)
    while i < n:
        if activations[i] > 0:
            span_tokens = [tokens[i]]
            i += 1
            while i < n and activations[i] > 0:
                span_tokens.append(tokens[i])
                i += 1
            highlighted.append("<<{}>>".format("".join(span_tokens)))
        else:
            highlighted.append(tokens[i])
            i += 1

    highlighted_str = "".join(highlighted).strip()

    # Only show non-zero activations
    activation_pairs = [
        f"({repr(t)}, {int(100 * a)})"
        for t, a in zip(tokens, activations)
        if a > 0
    ]
    activation_str = ", ".join(activation_pairs)

    return f"Text: {highlighted_str}\nActivations: {activation_str}"


def rerun_auto_interpretation(feature_id, use_closed_source: bool = True):
    if feature_id is None:
        return "[unlabeled]"
    
    link = f"https://transformer-circuits.pub/2025/attribution-graphs/features/jackl-circuits-runs-1-4-sofa-v3_0/{feature_id}.json"

    try:
        response = requests.get(link)
        data = response.json()
        toplogits = data.get("top_logits", None)
        # print(f"\n----> Rerunning interpretation for feature {feature_id} | toplogits: {toplogits}\n")

        # Gather top activating examples (up to 15), skipping ones with no activations
        examples_raw = data.get("examples_quantiles", [])
        top_examples = []
        if examples_raw:
            for ex in examples_raw[0].get("examples", []):
                formatted = format_activating_example(ex["tokens"], ex["tokens_acts_list"])
                if formatted is not None:
                    top_examples.append(formatted)

        has_logits = toplogits is not None and len(toplogits) > 0
        has_examples = len(top_examples) > 0

        if not has_logits and not has_examples:
            return "[unlabeled]"

        # Build prompt
        prompt_parts = [
            "Your job is to identify patterns in text. You are given information about a feature in a neural network "
            "and must produce a concise *Representative Term* that best captures what this feature represents.\n\n"

            "Use the provided information — top next tokens (tokens the feature boosts during prediction) and top activating text examples (examples that cause the feature to fire up) — "
            "to infer the feature's meaning. Weight whichever source shows a clearer, more coherent pattern more heavily.\n\n"

            "Features generally fall into two categories:\n"
            "- **Concept features**: The top activating examples share a coherent semantic pattern. Label these based on that pattern "
            "(e.g. 'legal disclaimers', 'French cooking terms', 'capital cities'). Aim for this level of specificity — "
            "not 'positive language' but 'enthusiastic product reviews'.\n"
            "- **Token-promoting features**: The top next tokens clearly converge on a specific token. Label these as 'say [token]' "
            "(e.g. 'say the', 'say Paris').\n\n"
            "Use 'say [token]' only when the activating examples lack a clear semantic pattern and token promotion is the dominant signal.\n"

            "If no clear pattern emerges across examples, or the feature appears to activate on unrelated concepts, "
            "use 'unclear' or 'polysemantic: [X] and [Y]' as the label.\n\n"

            "The context window around each highlighted <<span>> often reveals the true trigger — "
            "a feature may appear to activate on a noun but actually tracks the preceding verb or surrounding syntactic context. "
            "Examine what precedes and follows each highlighted span carefully.\n\n"

            "Guidelines:\n"
            "- The label must be a single short phrase — specific, not generic.\n"
            "- Do not include any explanation outside the JSON object.\n\n"
        ]

        if has_logits:
            prompt_parts.append("\n\n--------------------\n\n")
            prompt_parts.append(f"Top next tokens (tokens this feature promotes):\n{' || '.join(toplogits)}\n\n")
            prompt_parts.append("--------------------\n\n")

        if has_examples:
            prompt_parts.append(
                "Below are top activating examples (tokens inside <<>> are where this feature activates most strongly, "
                "with activation percentages shown). Pay attention to a few tokens before and after each highlighted span "
                "to understand what triggers this feature.\n\nTop activating examples:\n\n"
            )
            prompt_parts.append("\n\n".join(top_examples))
            prompt_parts.append("\n\n--------------------\n\n")

        prompt_parts.append(
            "Respond ONLY with a single valid JSON object — no prose, no markdown fences, nothing else.\n"
            "The JSON must have exactly these two fields:\n"
            '  "reasoning": a brief explanation of the pattern you identified (1-3 sentences),\n'
            '  "label": the concise representative term (single short phrase).\n\n'
            'Example: {"reasoning": "All examples show currency amounts in financial contexts.", "label": "monetary values in financial text"}\n'
        )

        prompt = "".join(prompt_parts)

        def _call_and_parse_interp(p):
            raw = generate_text(p, max_tokens=2048, temperature=1.0, use_closed_source=use_closed_source)
            # print(f"--DEBUG: RESPONSE:\n{raw}")
            return parse_json_response(raw, "label")

        new_label = _call_and_parse_interp(prompt)

        # Retry with a stricter prompt if parsing failed
        if not new_label:
            print(f"  [retry] JSON parse failed for feature {feature_id}, retrying with strict prompt.")
            retry_prompt = (
                prompt
                + "\nIMPORTANT: Your previous response could not be parsed. "
                "You MUST respond with ONLY a raw JSON object, e.g.: "
                '{"reasoning": "...", "label": "..."}\n'
            )
            new_label = _call_and_parse_interp(retry_prompt)

        if new_label:
            print(f"\n~~~~> Rerun interpretation for feature {feature_id} | new label: {new_label} | Top logits: {toplogits}\n")
            return new_label
        elif has_logits:
            print(f"  [fallback] Using top logit for feature {feature_id}.")
            return toplogits[0]
        else:
            return "[unlabeled]"
        
    except Exception as e:
        print(f"Error fetching interpretation for feature {feature_id}: {e}")
        return "[unlabeled]"
    
    
def get_umbrella_term(phrases, use_closed_source: bool = True):
    # Qwen/Qwen3-32B
    # meta-llama/Llama-3.3-70B-Instruct
    print(f"\n----> Generating supernode label | {phrases}")
    # phrases = [p.replace("Emb: ", "") for p in phrases]
    joined_phrases = '\n\n'.join(phrases)
    if joined_phrases.isspace():
        print("\tReturning None.")
        return "None"
    

    prompt = (
        "You are given several feature descriptions from nodes in a computational graph. "
        "Your task is to produce one concise *Representative Term* that best represents their *shared functional meaning*.\n\n"
        "Rules:\n"
        "- If all features are essentially the same, return that term (or a minimal cleaned version).\n"
        "- Do not choose a term solely because it is the most common literal token.\n"
        "- Identify the functional role or operational behavior the labels describe, not just the surface words.\n"
        "- Avoid returning a term that merely repeats a shared token unless that token actually is the shared meaning.\n"
        "- The term must be a single short phrase — specific, not generic.\n\n"
        f"Node features:\n{joined_phrases}\n\n"
        "Respond ONLY with a single valid JSON object — no prose, no markdown fences, nothing else.\n"
        "The JSON must have exactly these two fields:\n"
        '  "reasoning": a brief explanation of why this term fits (1-2 sentences),\n'
        '  "term": the concise representative term (single short phrase).\n\n'
        'Example: {"reasoning": "All features relate to predicting closing punctuation after a clause.", "term": "clause-final punctuation prediction"}\n'
    )

    def _call_and_parse_umbrella(p):
        raw = generate_text(p, max_tokens=2048, temperature=1.0, use_closed_source=use_closed_source)
        return parse_json_response(raw, "term")

    umbrella_label = _call_and_parse_umbrella(prompt)

    # Retry with stricter prompt if parsing failed
    if not umbrella_label:
        print("  [retry] JSON parse failed for umbrella term, retrying with strict prompt.")
        retry_prompt = (
            prompt
            + "\nIMPORTANT: Your previous response could not be parsed. "
            "You MUST respond with ONLY a raw JSON object, e.g.: "
            '{"reasoning": "...", "term": "..."}\n'
        )
        umbrella_label = _call_and_parse_umbrella(retry_prompt)

    if not umbrella_label:
        print("  [fallback] Could not extract umbrella term after retry.")
        umbrella_label = "[unlabeled]"

    print(f"{umbrella_label=}")
    print()
    return umbrella_label


def get_feature_top_examples_Haiku(feature_id, top_k=3):
    link = f"https://transformer-circuits.pub/2025/attribution-graphs/features/jackl-circuits-runs-1-4-sofa-v3_0/{feature_id}.json"

    response = requests.get(link)
    data = response.json()
    top_activating_examples = data.get("examples_quantiles", [])[0]
    examples = top_activating_examples["examples"]
    top_k = min(top_k, len(examples))
    examples = examples[:top_k]

    example_list = []
    for ex in examples:
        highlighted_str, activation_str = highlight_tokens(ex["tokens"], ex["tokens_acts_list"])
        example_list.append({
            "highlighted_str": highlighted_str,
            "activations_str": activation_str
        })
    
    return example_list


def highlight_tokens(tokens, activations):
    """
    tokens: list of strings
    activations: list of floats (same length as tokens)

    Returns:
        highlighted_str: tokens concatenated, with << >> around consecutive non-zero activation spans
        activation_str: string with (token, activation) tuples
    """

    assert len(tokens) == len(activations), "Lists must have same length."

    highlighted = []
    i = 0
    n = len(tokens)

    while i < n:
        if activations[i] != 0:
            # Start a span
            span_tokens = [tokens[i]]
            i += 1
            # Collect consecutive activated tokens
            while i < n and activations[i] != 0:
                span_tokens.append(tokens[i])
                i += 1
            highlighted.append(" <<{}>> ".format("".join(span_tokens)))
        else:
            highlighted.append(tokens[i])
            i += 1

    highlighted_str = "".join(highlighted)
    highlighted_str = highlighted_str.replace("  ", " ").strip()

    # Build activation output
    activation_pairs = []
    for t, a in zip(tokens, activations):
        activation_pairs.append(f"({repr(t)}, {int(100*a)})")
    activation_str = ", ".join(activation_pairs)

    return highlighted_str, activation_str


# exs = get_feature_top_examples_Haiku(feature_id="12505127", top_k=3)
# print(exs)
