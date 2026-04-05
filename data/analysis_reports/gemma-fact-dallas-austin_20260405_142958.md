# Circuit Analysis Report

**Generated:** 2026-04-05 14:29:58  
**Vignette:** `data/outputs/Gemma/gemma-fact-dallas-austin_skeleton.json`  
**Model (scan):** `gemma-2-2b`  
**Analysis LLM:** `gpt-5.4`  
**Slug:** `gemma-fact-dallas-austin`  
**Elapsed:** 25.4s  

---

## Circuit Prompt

```
<bos>Fact: The capital of the state containing Dallas is
```

## Task Description

Factual/reasoning task: '<bos>Fact: The capital of the state containing Dallas is'

## Counterfactual Task

Not specified

---

## Full Analysis Prompt Sent to LLM

<details>
<summary>Click to expand full prompt</summary>

```
You are given a sparse feature circuit extracted from a language model in JSON format.

## Task context
- **Model:** gemma-2-2b
- **Input prompt:** `<bos>Fact: The capital of the state containing Dallas is`
- **Task description:** Factual/reasoning task: '<bos>Fact: The capital of the state containing Dallas is'
- **Counterfactual task (if any):** Not specified

## Circuit summary
The circuit has been skeletonized into supernodes (semantic clusters of features).
Supernodes:
  - **Output " Austin" (p=0.450)** → nodes: 27_22605_10
  - **Texas Legal Information** → nodes: 14_2268_9, 16_25_9
  - **Geographic Place Names** → nodes: 20_15589_10, 21_5943_10, 23_12237_10, E_26865_9
  - **in** → nodes: 24_6044_10
  - **Punctuation Markers** → nodes: E_2_0, E_18143_1, E_235292_2, E_714_3
  - **Sentence Structure Elements** → nodes: E_6037_4, E_576_5, E_2329_7, E_10751_8, E_603_10, 27_780_10
  - **Output " the" (p=0.055)** → nodes: 27_573_10

Top influential nodes (by activation × influence):
  - `0_437_1` (layer 0) | influence=0.000 activation=0.000 | "Metadata and associative terms"
  - `0_478_1` (layer 0) | influence=0.000 activation=0.000 | "code syntax"
  - `0_997_1` (layer 0) | influence=0.000 activation=0.000 | "facts"
  - `0_1847_1` (layer 0) | influence=0.000 activation=0.000 | "crystallization"
  - `0_1903_1` (layer 0) | influence=0.000 activation=0.000 | "The"
  - `0_2115_1` (layer 0) | influence=0.000 activation=0.000 | "%\]"
  - `0_2616_1` (layer 0) | influence=0.000 activation=0.000 | "already existing"
  - `0_2724_1` (layer 0) | influence=0.000 activation=0.000 | "simplicity or ease"
  - `0_3461_1` (layer 0) | influence=0.000 activation=0.000 | "experiences"
  - `0_4175_1` (layer 0) | influence=0.000 activation=0.000 | "grand"
  - `0_4254_1` (layer 0) | influence=0.000 activation=0.000 | "pretty"
  - `0_4823_1` (layer 0) | influence=0.000 activation=0.000 | "technical/scientific text"
  - `0_4956_1` (layer 0) | influence=0.000 activation=0.000 | "time"
  - `0_4993_1` (layer 0) | influence=0.000 activation=0.000 | "Stage or Period"
  - `0_5215_1` (layer 0) | influence=0.000 activation=0.000 | "code"

## Full circuit JSON
```json
{
  "metadata": {
    "scan": "gemma-2-2b",
    "prompt": "<bos>Fact: The capital of the state containing Dallas is",
    "slug": "gemma-fact-dallas-austin"
  },
  "supernodes": [
    [
      "Output \" Austin\" (p=0.450)",
      "27_22605_10"
    ],
    [
      "Texas Legal Information",
      "14_2268_9",
      "16_25_9"
    ],
    [
      "Geographic Place Names",
      "20_15589_10",
      "21_5943_10",
      "23_12237_10",
      "E_26865_9"
    ],
    [
      "in",
      "24_6044_10"
    ],
    [
      "Punctuation Markers",
      "E_2_0",
      "E_18143_1",
      "E_235292_2",
      "E_714_3"
    ],
    [
      "Sentence Structure Elements",
      "E_6037_4",
      "E_576_5",
      "E_2329_7",
      "E_10751_8",
      "E_603_10",
      "27_780_10"
    ],
    [
      "Output \" the\" (p=0.055)",
      "27_573_10"
    ]
  ],
  "nodes": [
    {
      "node_id": "0_437_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Metadata and associative terms",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_478_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "code syntax",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_997_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "facts",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_1847_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "crystallization",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_1903_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "The",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_2115_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "%\\]",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_2616_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "already existing",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_2724_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "simplicity or ease",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_3461_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "experiences",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_4175_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "grand",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_4254_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "pretty",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_4823_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "technical/scientific text",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_4956_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "time",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_4993_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Stage or Period",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_5215_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "code",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_5369_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Names",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_5590_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Mobile phones, scripts",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_6078_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "What",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_6116_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "rare or formal language",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_6133_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "opinions and reviews",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_6421_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "work, projects, official activities",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_7344_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Fig",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_7376_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "writer",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_7465_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "aspects",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_7692_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "decrease",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_7812_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "raise",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_8008_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "she",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_8098_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "cost savings",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_8163_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "economics, politics, legal/science",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_8489_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "experiment findings",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_8540_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "unique",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_8642_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "actually",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_9026_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "(\\<",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_9637_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Comp",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_9689_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "computer code and website design",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_10138_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "comments",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_10317_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": " single character variables and parts of words that look like names or list-like data structures",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_10345_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "picture",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_11577_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "trails or conservatism",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_11670_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "higher",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_11888_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "straight",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_11938_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "facts",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_12186_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Federal",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_12200_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Scientific studies and research",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_12312_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "technical specifications",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_12601_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "fact",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_12605_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "temporal context",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_12698_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "questions and code",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_12914_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "1",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_13145_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "emotions",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_13344_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "transition words",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_14244_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "static",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_14725_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "code",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_16170_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "scientific/technical texts",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_16187_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "scope",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_268_2",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": ":",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_2858_2",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "colon",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_4325_2",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": ":",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_4825_2",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": " angle brackets and other miscellaneous markup tokens.",
      "influence": 0,
      "activation": 0
    },
    {
      "node_id": "0_4965_2",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": ":",
      "influence": 0,
      "activation": 0
    }
  ],
  "links": []
}
```

## Your analysis tasks

1. **Semantic clusters** — Identify the main semantic clusters of nodes and what concept each represents.
2. **Computational pathway** — Trace the main activation path from input tokens to the output token.
3. **High-influence & bottleneck nodes** — Which nodes have the highest influence? Are there bottlenecks?
4. **Circuit function** — What is this circuit doing? Is it factual recall, arithmetic reasoning, instruction following, formatting, or something else?
5. **Predicted output** — What output was the model generating when this circuit was active? Why?
6. **Ablation prediction** — What would happen if the top-influence node were ablated?
7. **Generalization prediction** — Based on this circuit, predict whether the model will generalize to the counterfactual task. Explain your reasoning in terms of which circuit components would or would not transfer.
8. **Reasoning vs. memorization** — Does this circuit reflect robust algorithmic reasoning or surface-level pattern matching / memorization? Justify with specific nodes.

```

</details>

---

## LLM Analysis

1. **Semantic clusters**

The provided skeletonized circuit is dominated by a few coarse supernodes:

- **Output " Austin"**  
  - Node: `27_22605_10`  
  - This is the main answer token node and is the strongest output candidate in the summary (`p=0.450`).

- **Texas Legal Information**  
  - Nodes: `14_2268_9`, `16_25_9`  
  - This suggests the model is using a Texas-associated factual subspace, though the label “legal information” implies the feature may be broader Texas/government/jurisdiction material rather than a clean “Texas” identity feature.

- **Geographic Place Names**  
  - Nodes: `20_15589_10`, `21_5943_10`, `23_12237_10`, `E_26865_9`  
  - These likely respond to named places such as “Dallas,” “state,” and possibly state/capital relations in general.

- **in**  
  - Node: `24_6044_10`  
  - A relational / syntactic feature, possibly involved in the phrase structure “capital of the state containing Dallas”.

- **Punctuation Markers**  
  - Nodes: `E_2_0`, `E_18143_1`, `E_235292_2`, `E_714_3`  
  - These likely track formatting from `<bos>` and `Fact:`. They are support features, not the substantive knowledge-bearing content.

- **Sentence Structure Elements**  
  - Nodes: `E_6037_4`, `E_576_5`, `E_2329_7`, `E_10751_8`, `E_603_10`, `27_780_10`  
  - These appear to represent generic prompt-completion structure and help the model continue a factual sentence in the expected grammatical form.

- **Output " the"**  
  - Node: `27_573_10`  
  - A weaker competing continuation, probably reflecting generic sentence-completion uncertainty.

Overall, the main semantic content is a combination of:
- a **Texas/government fact cluster**
- a **place-name cluster**
- a **syntactic relation cluster**
- and a **fact-statement / formatting scaffold**

2. **Computational pathway**

The JSON does not include explicit links, so the pathway has to be inferred from the supernode structure rather than traced edge-by-edge.

Most likely pathway:

- The input prompt contains:
  - `Fact:` → activates factual-statement / declarative-answer framing
  - `The capital of the state containing Dallas` → activates:
    - place-name features for **Dallas**
    - geographic/jurisdiction features for **state**
    - capital-related factual retrieval structure
- These combine into a latent representation of:
  - “Dallas is in Texas”
  - “capital of Texas is Austin”
- That latent state then supports the output node **`27_22605_10` = " Austin"`**

So the broad computation appears to be:

**Prompt formatting + factual frame**  
→ **recognition of a geographic entity (“Dallas”)**  
→ **mapping from Dallas to Texas**  
→ **mapping from Texas to its capital**  
→ **token emission: “ Austin”**

However, the circuit as given does **not** isolate two clearly separable reasoning hops with explicit intermediate Texas and Austin edges. Instead, the supernodes suggest a compressed factual-association pathway rather than a fully transparent multi-step chain.

3. **High-influence & bottleneck nodes**

The data here is unusually sparse/degenerate:

- The “top influential nodes” list consists entirely of layer-0 nodes with:
  - `influence = 0`
  - `activation = 0`

So from the supplied node table, there are **no measurable high-influence nodes** in the usual sense.

What this means analytically:
- The real influential nodes are likely the supernode members listed in the summary, but their individual numeric values are not present in the node table.
- Therefore the likely practical bottlenecks are:
  - **`27_22605_10`** — final output node for `" Austin"`
  - the **Texas Legal Information** supernode
  - the **Geographic Place Names** supernode

If forced to identify conceptual bottlenecks, they are:
- the latent representation that identifies **Dallas → Texas**
- the latent representation that supports **Texas → Austin**

But the circuit dump does not expose those as individually quantified nodes.

4. **Circuit function**

This circuit is primarily doing **factual recall / associative retrieval**, with light compositional structure.

It is not:
- arithmetic
- instruction following
- formatting-only behavior

It is closest to:
- **geographic factual QA**
- possibly a shallow two-hop relation:
  - find the state containing Dallas
  - retrieve that state’s capital

But the circuit evidence favors **compressed factual association** over explicit symbolic reasoning. The presence of:
- **Texas Legal Information**
- **Geographic Place Names**
- factual prompt structure

suggests the model is activating stored knowledge around Texas and then emitting Austin.

5. **Predicted output**

The active output is:

- **`" Austin"` with probability 0.450**

Why:
- The prompt asks for the capital of the state containing Dallas.
- Dallas is in Texas.
- Austin is the capital of Texas.
- The circuit contains exactly the kinds of clusters expected for that retrieval:
  - Texas-related knowledge
  - place names
  - sentence-completion scaffolding
- The weaker competing output `" the"` likely reflects generic continuation pressure, but the main answer node is clearly `" Austin"`.

6. **Ablation prediction**

The prompt asks about the **top-influence node**, but the listed top-influence nodes all have zero influence and zero activation, so ablating any one of those listed nodes should have **no effect** according to the provided data.

If instead we interpret “top-influence node” as the main answer-bearing node:
- Ablating **`27_22605_10`** would strongly reduce probability of `" Austin"`.
- The model would likely fall back to:
  - another generic token like `" the"`
  - or another location-related continuation if alternative answer pathways exist outside this extracted circuit.

If ablating the conceptual Texas bottleneck:
- The model would likely lose the Dallas→Texas→Austin mapping and either become uncertain or produce a wrong continuation.

7. **Generalization prediction**

There is no specific counterfactual task given, but based on this circuit I would predict:

- **Good transfer to nearby paraphrases**
  - e.g. “Dallas is located in which state’s capital city is…”
  - or “The capital of the state that contains Dallas is…”
- **Weaker transfer to true counterfactuals or less familiar entities**
  - e.g. replacing Dallas with a less common city
  - or asking for a novel geographic relation requiring a more explicit compositional chain

Reason:
- The circuit components seem fairly **entity- and fact-specific**:
  - Texas-related supernode
  - place-name supernode
- There is only weak evidence for a robust general “city → state → capital” algorithm.
- The extracted features look more like stored associations triggered by the prompt than a reusable relational program.

So:
- **Paraphrase generalization:** likely yes
- **Counterfactual or out-of-distribution transfer:** limited, unless the model already memorized the corresponding facts in similar feature clusters

8. **Reasoning vs. memorization**

This circuit looks more like **surface-level factual association / memorization** than robust algorithmic reasoning.

Evidence:

- The semantic clusters are highly content-specific:
  - **Texas Legal Information**
  - **Geographic Place Names**
- There is no explicit chain of interpretable intermediate nodes corresponding cleanly to:
  - identify Dallas as a city
  - infer Texas as containing state
  - retrieve Austin as capital
- The absence of visible high-influence intermediate links suggests the extracted circuit is not exposing a strongly modular reasoning chain.
- The prompt framing `Fact:` and sentence-structure nodes suggest the model is in a “complete a known fact” mode.

That said, it is not purely one-step memorization of the exact string:
- The prompt is slightly indirect (“state containing Dallas” rather than “Texas”).
- So some compositionality is likely present.

Best characterization:
- **shallow compositional factual recall**
- not robust algorithmic reasoning
- not mere exact-string lookup either

In short: the circuit appears to retrieve a familiar geographic fact through a small amount of relational framing, but the evidence supports **memorized knowledge organized by semantic associations**, not a general-purpose reasoning algorithm.

---

_Report generated by `experiments/analyze_vignette.py`_