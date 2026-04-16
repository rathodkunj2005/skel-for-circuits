# Circuit Analysis Report

**Generated:** 2026-04-13 20:08:18  
**Vignette:** `data/sample_graphs/Gemma/gemma-fact-dallas-austin.json`  
**Model (scan):** `gemma-2-2b`  
**Analysis LLM:** `gpt-5.4`  
**Slug:** `gemma-fact-dallas-austin`  
**Elapsed:** 46.0s total  

---

## Circuit Prompt

```
<bos>Fact: The capital of the state containing Dallas is
```

## Task Description

Two-step reasoning based on geography. For example, a prompt could look like:
Fact: The capital of the state containing Dallas is

## Counterfactual Task

Two-step reasoning based on geography, but with an alternate world. For example, a prompt could look like:
In this world, Boise is the capital of Texas, Salt Lake City is the capital of Vermont, Raleigh is the capital of Georgia, ...
Fact: The capital of the state containing Dallas is

---

## Model-Only Analysis

**Elapsed:** 8.9s  

### Full Analysis Prompt Sent to LLM

<details>
<summary>Click to expand full prompt</summary>

```
You are evaluating whether a language model is likely to generalize from one task description to another.

## Task context
- **Test model being evaluated:** gemma-2-2b
- **Input prompt:** `<bos>Fact: The capital of the state containing Dallas is`
- **Default task description:** Two-step reasoning based on geography. For example, a prompt could look like:
Fact: The capital of the state containing Dallas is
- **Counterfactual task to predict generalization for:** Two-step reasoning based on geography, but with an alternate world. For example, a prompt could look like:
In this world, Boise is the capital of Texas, Salt Lake City is the capital of Vermont, Raleigh is the capital of Georgia, ...
Fact: The capital of the state containing Dallas is

## Your analysis tasks

1. **Default-task capability hypothesis** — Based on the prompt and task description alone, what capability does the model appear to need?
2. **Counterfactual transfer prediction** — Would that capability likely transfer to the counterfactual task?
3. **Reasoning vs. memorization prior** — Is success on the default task more likely to rely on reasoning, memorization, or a mix?
4. **Uncertainty** — What key uncertainty limits your prediction, given that you are not shown the model's internal circuit?

## Final verdict (REQUIRED — answer this last, after your analysis)

Based only on the task descriptions and prompt context, predict whether the test model will correctly answer instances of the counterfactual task.

End your response with exactly this format:
```
GENERALIZATION VERDICT: [YES or NO]
CONFIDENCE: [integer 0-100]%
KEY EVIDENCE: [one sentence explaining the main reason for this prediction]
```

```

</details>

### LLM Analysis

1. **Default-task capability hypothesis**  
The model appears to need a simple **two-hop geographic retrieval/composition** capability:
- identify that **Dallas is in Texas**
- retrieve that **the capital of Texas is Austin**
- output **Austin**

So the task is not just direct recall of “capital of Dallas’s state,” but composition over two known relations: **city → state** and **state → capital**.

2. **Counterfactual transfer prediction**  
Probably **not**. The counterfactual version requires the model to:
- read and temporarily store a set of **alternate-world overrides**
- use those overrides instead of strong real-world associations
- still perform the same two-step composition under the overridden mapping

That is a stricter capability than ordinary geography QA. Many models can do the real-world version via entrenched knowledge, but fail when they must suppress that knowledge and substitute prompt-given fictional facts.

3. **Reasoning vs. memorization prior**  
Success on the default task is most likely a **mix, leaning heavily on memorized knowledge**.  
The “reasoning” required is shallow composition, but both needed facts are common world knowledge likely stored directly in the model. A model could succeed either by chaining two facts or by pattern-completing from familiar associations around “Dallas,” “Texas,” and “Austin.”

4. **Uncertainty**  
The key uncertainty is whether the model has a robust **in-context override mechanism** for factual knowledge. Without seeing the circuit, we do not know whether its behavior is driven by:
- flexible relational composition that can bind to prompt-provided mappings, or
- rigid retrieval of canonical geography facts

That distinction is exactly what determines counterfactual success.

GENERALIZATION VERDICT: NO
CONFIDENCE: 78%
KEY EVIDENCE: The counterfactual task requires overriding strong memorized geography facts with prompt-specified alternate mappings, which models often fail to do reliably even when they can answer the default real-world question.

---

## Circuit-Augmented Analysis

**Elapsed:** 37.1s  

### Full Analysis Prompt Sent to LLM

<details>
<summary>Click to expand full prompt</summary>

```
You are given a sparse feature circuit extracted from a language model in JSON format.

## Task context
- **Test model (whose circuit this is):** gemma-2-2b
- **Input prompt the circuit was extracted from:** `<bos>Fact: The capital of the state containing Dallas is`
- **Default task description:** Two-step reasoning based on geography. For example, a prompt could look like:
Fact: The capital of the state containing Dallas is
- **Counterfactual task to predict generalization for:** Two-step reasoning based on geography, but with an alternate world. For example, a prompt could look like:
In this world, Boise is the capital of Texas, Salt Lake City is the capital of Vermont, Raleigh is the capital of Georgia, ...
Fact: The capital of the state containing Dallas is

## Circuit summary
The circuit has been skeletonized into supernodes (semantic clusters of features).
Supernodes (each is a semantically coherent group of nodes):
  - **capital** → nodes: 15_4494_4, 6_4662_4, 4_7671_4, 3_13984_4, 1_1000_4
  - **state** → nodes: 6_4012_7, 0_13727_7
  - **Texas** → nodes: 20_15589_9, 19_7477_9, 16_25_9, 4_13154_9, 14_2268_9, 7_6861_9
  - **preposition followed by place name** → nodes: 19_1445_10, 18_6101_10
  - **capital cities / say a capital city** → nodes: 21_5943_10, 17_7178_10, 7_691_10, 16_4298_10

Key pinned nodes (highest-influence nodes selected by the skeletonization pipeline):
  - `27_22605_10` (layer 27) | influence=0.000 activation=0.000 | "Output " Austin" (p=0.450)"
  - `20_15589_10` (layer 20) | influence=0.258 activation=49.562 | "Texas"
  - `E_26865_9` (layer E) | influence=0.145 activation=0.000 | "Emb: "  Dallas""
  - `21_5943_10` (layer 21) | influence=0.297 activation=56.370 | "cities"
  - `23_12237_10` (layer 23) | influence=0.302 activation=54.440 | "Cities and states names"
  - `20_15589_9` (layer 20) | influence=0.326 activation=45.661 | "Texas"
  - `16_25_9` (layer 16) | influence=0.315 activation=28.161 | "Texas legal documents"
  - `14_2268_9` (layer 14) | influence=0.311 activation=25.700 | "Texas"
  - `18_8959_10` (layer 18) | influence=0.356 activation=39.965 | "government/state"
  - `4_13154_9` (layer 4) | influence=0.341 activation=15.166 | "Texas"
  - `7_6861_9` (layer 7) | influence=0.354 activation=15.515 | "Texas legal cases"
  - `19_1445_10` (layer 19) | influence=0.382 activation=35.446 | "Downtowns of cities"
  - `E_2329_7` (layer E) | influence=0.213 activation=0.000 | "Emb: "  state""
  - `E_6037_4` (layer E) | influence=0.042 activation=0.000 | "Emb: "  capital""
  - `0_13727_7` (layer 0) | influence=0.503 activation=9.575 | "State"

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
      "capital",
      "15_4494_4",
      "6_4662_4",
      "4_7671_4",
      "3_13984_4",
      "1_1000_4"
    ],
    [
      "state",
      "6_4012_7",
      "0_13727_7"
    ],
    [
      "Texas",
      "20_15589_9",
      "19_7477_9",
      "16_25_9",
      "4_13154_9",
      "14_2268_9",
      "7_6861_9"
    ],
    [
      "preposition followed by place name",
      "19_1445_10",
      "18_6101_10"
    ],
    [
      "capital cities / say a capital city",
      "21_5943_10",
      "17_7178_10",
      "7_691_10",
      "16_4298_10"
    ]
  ],
  "nodes": [
    {
      "node_id": "27_22605_10",
      "layer": "27",
      "feature_type": "logit",
      "clerp": "Output \" Austin\" (p=0.450)",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "20_15589_10",
      "layer": "20",
      "feature_type": "cross layer transcoder",
      "clerp": "Texas",
      "influence": 0.2576,
      "activation": 49.5624,
      "pinned": true
    },
    {
      "node_id": "E_26865_9",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \"  Dallas\"",
      "influence": 0.145,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "21_5943_10",
      "layer": "21",
      "feature_type": "cross layer transcoder",
      "clerp": "cities",
      "influence": 0.2973,
      "activation": 56.3697,
      "pinned": true
    },
    {
      "node_id": "23_12237_10",
      "layer": "23",
      "feature_type": "cross layer transcoder",
      "clerp": "Cities and states names",
      "influence": 0.302,
      "activation": 54.4402,
      "pinned": true
    },
    {
      "node_id": "20_15589_9",
      "layer": "20",
      "feature_type": "cross layer transcoder",
      "clerp": "Texas",
      "influence": 0.3256,
      "activation": 45.6607,
      "pinned": true
    },
    {
      "node_id": "16_25_9",
      "layer": "16",
      "feature_type": "cross layer transcoder",
      "clerp": "Texas legal documents",
      "influence": 0.3147,
      "activation": 28.1611,
      "pinned": true
    },
    {
      "node_id": "14_2268_9",
      "layer": "14",
      "feature_type": "cross layer transcoder",
      "clerp": "Texas",
      "influence": 0.3108,
      "activation": 25.6999,
      "pinned": true
    },
    {
      "node_id": "18_8959_10",
      "layer": "18",
      "feature_type": "cross layer transcoder",
      "clerp": "government/state",
      "influence": 0.3561,
      "activation": 39.9646,
      "pinned": true
    },
    {
      "node_id": "4_13154_9",
      "layer": "4",
      "feature_type": "cross layer transcoder",
      "clerp": "Texas",
      "influence": 0.3408,
      "activation": 15.1665,
      "pinned": true
    },
    {
      "node_id": "7_6861_9",
      "layer": "7",
      "feature_type": "cross layer transcoder",
      "clerp": "Texas legal cases",
      "influence": 0.3539,
      "activation": 15.5151,
      "pinned": true
    },
    {
      "node_id": "19_1445_10",
      "layer": "19",
      "feature_type": "cross layer transcoder",
      "clerp": "Downtowns of cities",
      "influence": 0.3825,
      "activation": 35.4459,
      "pinned": true
    },
    {
      "node_id": "E_2329_7",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \"  state\"",
      "influence": 0.213,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "E_6037_4",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \"  capital\"",
      "influence": 0.0422,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "0_13727_7",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "State",
      "influence": 0.5027,
      "activation": 9.5754,
      "pinned": true
    },
    {
      "node_id": "6_4012_7",
      "layer": "6",
      "feature_type": "cross layer transcoder",
      "clerp": "state abbreviations",
      "influence": 0.4855,
      "activation": 14.2638,
      "pinned": true
    },
    {
      "node_id": "17_7178_10",
      "layer": "17",
      "feature_type": "cross layer transcoder",
      "clerp": "government buildings",
      "influence": 0.4356,
      "activation": 27.4991,
      "pinned": true
    },
    {
      "node_id": "15_4494_4",
      "layer": "15",
      "feature_type": "cross layer transcoder",
      "clerp": "capital",
      "influence": 0.4687,
      "activation": 68.6783,
      "pinned": true
    },
    {
      "node_id": "6_4662_4",
      "layer": "6",
      "feature_type": "cross layer transcoder",
      "clerp": "City",
      "influence": 0.5302,
      "activation": 15.0582,
      "pinned": true
    },
    {
      "node_id": "4_7671_4",
      "layer": "4",
      "feature_type": "cross layer transcoder",
      "clerp": "officer",
      "influence": 0.5293,
      "activation": 11.0963,
      "pinned": true
    },
    {
      "node_id": "3_13984_4",
      "layer": "3",
      "feature_type": "cross layer transcoder",
      "clerp": "capital",
      "influence": 0.555,
      "activation": 11.1554,
      "pinned": true
    },
    {
      "node_id": "1_1000_4",
      "layer": "1",
      "feature_type": "cross layer transcoder",
      "clerp": "capital",
      "influence": 0.3922,
      "activation": 24.3037,
      "pinned": true
    },
    {
      "node_id": "19_7477_9",
      "layer": "19",
      "feature_type": "cross layer transcoder",
      "clerp": "Dallas",
      "influence": 0.4591,
      "activation": 55.7774,
      "pinned": true
    },
    {
      "node_id": "18_6101_10",
      "layer": "18",
      "feature_type": "cross layer transcoder",
      "clerp": "capital cities",
      "influence": 0.3645,
      "activation": 20.9938,
      "pinned": true
    },
    {
      "node_id": "16_4298_10",
      "layer": "16",
      "feature_type": "cross layer transcoder",
      "clerp": "capital",
      "influence": 0.4407,
      "activation": 19.9297,
      "pinned": true
    },
    {
      "node_id": "7_691_10",
      "layer": "7",
      "feature_type": "cross layer transcoder",
      "clerp": "capital",
      "influence": 0.439,
      "activation": 16.3848,
      "pinned": true
    },
    {
      "node_id": "17_13250_7",
      "layer": "17",
      "feature_type": "cross layer transcoder",
      "clerp": "state",
      "influence": 0.6658,
      "activation": 73.1259,
      "pinned": false
    },
    {
      "node_id": "24_13277_10",
      "layer": "24",
      "feature_type": "cross layer transcoder",
      "clerp": "Romance languages",
      "influence": 0.332,
      "activation": 116.5678,
      "pinned": false
    },
    {
      "node_id": "10_14174_1",
      "layer": "10",
      "feature_type": "cross layer transcoder",
      "clerp": "Braces",
      "influence": 0.6171,
      "activation": 55.0935,
      "pinned": false
    },
    {
      "node_id": "17_7767_9",
      "layer": "17",
      "feature_type": "cross layer transcoder",
      "clerp": "US states",
      "influence": 0.6902,
      "activation": 48.4553,
      "pinned": false
    },
    {
      "node_id": "9_15819_1",
      "layer": "9",
      "feature_type": "cross layer transcoder",
      "clerp": "code",
      "influence": 0.6396,
      "activation": 50.3279,
      "pinned": false
    },
    {
      "node_id": "9_3056_1",
      "layer": "9",
      "feature_type": "cross layer transcoder",
      "clerp": " various keywords in programming languages, markup languages, and mathematical notations",
      "influence": 0.5562,
      "activation": 47.0232,
      "pinned": false
    },
    {
      "node_id": "24_5668_10",
      "layer": "24",
      "feature_type": "cross layer transcoder",
      "clerp": "Media channels",
      "influence": 0.6203,
      "activation": 37.7247,
      "pinned": false
    },
    {
      "node_id": "25_9975_10",
      "layer": "25",
      "feature_type": "cross layer transcoder",
      "clerp": " proper nouns",
      "influence": 0.5312,
      "activation": 42.0476,
      "pinned": false
    },
    {
      "node_id": "11_3544_9",
      "layer": "11",
      "feature_type": "cross layer transcoder",
      "clerp": "diabetic",
      "influence": 0.6798,
      "activation": 32.4265,
      "pinned": false
    },
    {
      "node_id": "20_5916_10",
      "layer": "20",
      "feature_type": "cross layer transcoder",
      "clerp": "capital",
      "influence": 0.4108,
      "activation": 53.4307,
      "pinned": false
    },
    {
      "node_id": "22_3064_10",
      "layer": "22",
      "feature_type": "cross layer transcoder",
      "clerp": "Texas/Dallas",
      "influence": 0.3681,
      "activation": 58.4977,
      "pinned": false
    },
    {
      "node_id": "25_3264_10",
      "layer": "25",
      "feature_type": "cross layer transcoder",
      "clerp": "capital",
      "influence": 0.6596,
      "activation": 31.5707,
      "pinned": false
    },
    {
      "node_id": "25_553_10",
      "layer": "25",
      "feature_type": "cross layer transcoder",
      "clerp": "general English text",
      "influence": 0.4064,
      "activation": 51.2128,
      "pinned": false
    },
    {
      "node_id": "22_3551_10",
      "layer": "22",
      "feature_type": "cross layer transcoder",
      "clerp": "Place names and legal cases",
      "influence": 0.4119,
      "activation": 48.3562,
      "pinned": false
    },
    {
      "node_id": "9_2762_1",
      "layer": "9",
      "feature_type": "cross layer transcoder",
      "clerp": "code snippets",
      "influence": 0.5907,
      "activation": 33.434,
      "pinned": false
    },
    {
      "node_id": "9_13483_1",
      "layer": "9",
      "feature_type": "cross layer transcoder",
      "clerp": "code, data, legal text",
      "influence": 0.682,
      "activation": 26.625,
      "pinned": false
    },
    {
      "node_id": "24_5999_10",
      "layer": "24",
      "feature_type": "cross layer transcoder",
      "clerp": "Technical text snippets",
      "influence": 0.4715,
      "activation": 36.9518,
      "pinned": false
    },
    {
      "node_id": "25_16302_10",
      "layer": "25",
      "feature_type": "cross layer transcoder",
      "clerp": "Legal/court cases",
      "influence": 0.4365,
      "activation": 39.7626,
      "pinned": false
    },
    {
      "node_id": "7_3099_9",
      "layer": "7",
      "feature_type": "cross layer transcoder",
      "clerp": "Acronyms or codes",
      "influence": 0.6347,
      "activation": 27.3042,
      "pinned": false
    },
    {
      "node_id": "8_3099_9",
      "layer": "8",
      "feature_type": "cross layer transcoder",
      "clerp": "code",
      "influence": 0.6211,
      "activation": 27.6912,
      "pinned": false
    },
    {
      "node_id": "16_15223_10",
      "layer": "16",
      "feature_type": "cross layer transcoder",
      "clerp": "say the ending \")\"",
      "influence": 0.5537,
      "activation": 30.0817,
      "pinned": false
    },
    {
      "node_id": "20_15276_9",
      "layer": "20",
      "feature_type": "cross layer transcoder",
      "clerp": "Dallas sports",
      "influence": 0.6775,
      "activation": 24.3917,
      "pinned": false
    },
    {
      "node_id": "25_4717_10",
      "layer": "25",
      "feature_type": "cross layer transcoder",
      "clerp": "unusual/fantastical narratives",
      "influence": 0.5175,
      "activation": 31.3252,
      "pinned": false
    },
    {
      "node_id": "16_12678_9",
      "layer": "16",
      "feature_type": "cross layer transcoder",
      "clerp": "cities",
      "influence": 0.4959,
      "activation": 32.0555,
      "pinned": false
    },
    {
      "node_id": "24_6044_10",
      "layer": "24",
      "feature_type": "cross layer transcoder",
      "clerp": "in",
      "influence": 0.2757,
      "activation": 55.0746,
      "pinned": false
    },
    {
      "node_id": "17_13508_9",
      "layer": "17",
      "feature_type": "cross layer transcoder",
      "clerp": "city names",
      "influence": 0.6439,
      "activation": 23.3799,
      "pinned": false
    },
    {
      "node_id": "25_12025_10",
      "layer": "25",
      "feature_type": "cross layer transcoder",
      "clerp": "broad, unrelated text",
      "influence": 0.6168,
      "activation": 24.0657,
      "pinned": false
    },
    {
      "node_id": "24_15627_10",
      "layer": "24",
      "feature_type": "cross layer transcoder",
      "clerp": "Locations",
      "influence": 0.5533,
      "activation": 26.4258,
      "pinned": false
    },
    {
      "node_id": "10_3837_10",
      "layer": "10",
      "feature_type": "cross layer transcoder",
      "clerp": "technical documents/publications",
      "influence": 0.6333,
      "activation": 22.4563,
      "pinned": false
    },
    {
      "node_id": "0_3820_6",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "the",
      "influence": 0.5109,
      "activation": 27.575,
      "pinned": false
    },
    {
      "node_id": "22_11220_10",
      "layer": "22",
      "feature_type": "cross layer transcoder",
      "clerp": "places and locations",
      "influence": 0.5185,
      "activation": 26.9848,
      "pinned": false
    },
    {
      "node_id": "7_462_1",
      "layer": "7",
      "feature_type": "cross layer transcoder",
      "clerp": "code snippets",
      "influence": 0.5664,
      "activation": 24.6555,
      "pinned": false
    },
    {
      "node_id": "7_5741_1",
      "layer": "7",
      "feature_type": "cross layer transcoder",
      "clerp": "Code snippets",
      "influence": 0.5975,
      "activation": 23.1014,
      "pinned": false
    },
    {
      "node_id": "25_4886_10",
      "layer": "25",
      "feature_type": "cross layer transcoder",
      "clerp": "last names",
      "influence": 0.4899,
      "activation": 27.9166,
      "pinned": false
    }
  ],
  "links": [
    {
      "source": "E_26865_9",
      "target": "19_7477_9",
      "weight": 68.8506
    },
    {
      "source": "E_2329_7",
      "target": "17_13250_7",
      "weight": 64.7495
    },
    {
      "source": "E_26865_9",
      "target": "20_15589_9",
      "weight": 62.2033
    },
    {
      "source": "E_6037_4",
      "target": "15_4494_4",
      "weight": 49.3197
    },
    {
      "source": "E_26865_9",
      "target": "20_15589_10",
      "weight": 43.259
    },
    {
      "source": "E_26865_9",
      "target": "22_3064_10",
      "weight": 37.9637
    },
    {
      "source": "E_26865_9",
      "target": "4_13154_9",
      "weight": 34.2273
    },
    {
      "source": "E_6037_4",
      "target": "25_3264_10",
      "weight": 32.8699
    },
    {
      "source": "E_6037_4",
      "target": "1_1000_4",
      "weight": 30.8149
    },
    {
      "source": "E_26865_9",
      "target": "16_25_9",
      "weight": 28.9526
    },
    {
      "source": "20_15589_10",
      "target": "24_6044_10",
      "weight": 28.6483
    },
    {
      "source": "E_26865_9",
      "target": "14_2268_9",
      "weight": 26.9398
    },
    {
      "source": "E_6037_4",
      "target": "3_13984_4",
      "weight": 26.4632
    },
    {
      "source": "E_26865_9",
      "target": "24_6044_10",
      "weight": 24.8152
    },
    {
      "source": "E_6037_4",
      "target": "6_4662_4",
      "weight": 23.8288
    },
    {
      "source": "16_25_9",
      "target": "20_15589_9",
      "weight": 22.1791
    },
    {
      "source": "E_26865_9",
      "target": "22_3551_10",
      "weight": 20.4768
    },
    {
      "source": "14_2268_9",
      "target": "16_25_9",
      "weight": 20.2962
    },
    {
      "source": "E_26865_9",
      "target": "7_6861_9",
      "weight": 20.1519
    },
    {
      "source": "14_2268_9",
      "target": "20_15589_9",
      "weight": 20.0184
    },
    {
      "source": "E_2329_7",
      "target": "18_8959_10",
      "weight": 19.6263
    },
    {
      "source": "E_6037_4",
      "target": "4_7671_4",
      "weight": 19.1083
    },
    {
      "source": "24_13277_10",
      "target": "25_16302_10",
      "weight": 18.9628
    },
    {
      "source": "9_3056_1",
      "target": "10_14174_1",
      "weight": 18.9269
    },
    {
      "source": "24_13277_10",
      "target": "25_9975_10",
      "weight": 18.7333
    },
    {
      "source": "20_15589_10",
      "target": "22_3064_10",
      "weight": 18.595
    },
    {
      "source": "24_5668_10",
      "target": "25_553_10",
      "weight": 18.224
    },
    {
      "source": "16_25_9",
      "target": "19_7477_9",
      "weight": 17.8132
    },
    {
      "source": "7_3099_9",
      "target": "8_3099_9",
      "weight": -16.6693
    },
    {
      "source": "24_13277_10",
      "target": "25_4717_10",
      "weight": 16.3983
    },
    {
      "source": "E_2329_7",
      "target": "6_4012_7",
      "weight": 16.3898
    },
    {
      "source": "14_2268_9",
      "target": "19_7477_9",
      "weight": 15.6179
    },
    {
      "source": "9_15819_1",
      "target": "10_14174_1",
      "weight": 15.5182
    },
    {
      "source": "17_7178_10",
      "target": "18_6101_10",
      "weight": 15.3246
    },
    {
      "source": "E_26865_9",
      "target": "16_12678_9",
      "weight": 15.1802
    },
    {
      "source": "E_26865_9",
      "target": "20_15276_9",
      "weight": 15.1113
    },
    {
      "source": "17_7178_10",
      "target": "18_8959_10",
      "weight": 14.3329
    },
    {
      "source": "14_2268_9",
      "target": "20_15589_10",
      "weight": 14.2462
    },
    {
      "source": "19_7477_9",
      "target": "20_15276_9",
      "weight": 14.0503
    },
    {
      "source": "18_6101_10",
      "target": "21_5943_10",
      "weight": 14.0175
    },
    {
      "source": "E_6037_4",
      "target": "19_1445_10",
      "weight": 13.5384
    },
    {
      "source": "4_13154_9",
      "target": "7_6861_9",
      "weight": 13.3737
    },
    {
      "source": "4_13154_9",
      "target": "20_15589_9",
      "weight": 13.2019
    },
    {
      "source": "7_3099_9",
      "target": "11_3544_9",
      "weight": -12.9241
    },
    {
      "source": "20_15589_10",
      "target": "22_3551_10",
      "weight": 12.8925
    },
    {
      "source": "21_5943_10",
      "target": "22_11220_10",
      "weight": 12.8296
    },
    {
      "source": "9_2762_1",
      "target": "10_14174_1",
      "weight": 12.6938
    },
    {
      "source": "8_3099_9",
      "target": "11_3544_9",
      "weight": -12.3885
    },
    {
      "source": "E_6037_4",
      "target": "16_4298_10",
      "weight": 12.3861
    },
    {
      "source": "E_26865_9",
      "target": "25_4717_10",
      "weight": 12.075
    },
    {
      "source": "E_2329_7",
      "target": "0_13727_7",
      "weight": 11.786
    },
    {
      "source": "7_462_1",
      "target": "9_15819_1",
      "weight": 11.1961
    },
    {
      "source": "7_6861_9",
      "target": "20_15589_9",
      "weight": 11.153
    },
    {
      "source": "22_3551_10",
      "target": "24_15627_10",
      "weight": 10.9228
    },
    {
      "source": "21_5943_10",
      "target": "23_12237_10",
      "weight": 10.8258
    },
    {
      "source": "18_8959_10",
      "target": "20_15589_10",
      "weight": 10.5845
    },
    {
      "source": "16_25_9",
      "target": "20_15589_10",
      "weight": 10.5184
    },
    {
      "source": "E_6037_4",
      "target": "17_7178_10",
      "weight": 10.1769
    },
    {
      "source": "19_7477_9",
      "target": "20_15589_9",
      "weight": -10.1437
    },
    {
      "source": "7_6861_9",
      "target": "16_25_9",
      "weight": 9.9463
    },
    {
      "source": "16_25_9",
      "target": "24_6044_10",
      "weight": 9.8284
    },
    {
      "source": "7_5741_1",
      "target": "9_15819_1",
      "weight": 9.788
    },
    {
      "source": "18_8959_10",
      "target": "20_5916_10",
      "weight": 9.7768
    },
    {
      "source": "17_7178_10",
      "target": "20_5916_10",
      "weight": 9.5956
    },
    {
      "source": "9_13483_1",
      "target": "10_14174_1",
      "weight": 9.4429
    },
    {
      "source": "21_5943_10",
      "target": "25_4717_10",
      "weight": -9.3903
    },
    {
      "source": "20_5916_10",
      "target": "25_3264_10",
      "weight": 9.3492
    },
    {
      "source": "4_13154_9",
      "target": "20_15589_10",
      "weight": 9.2187
    },
    {
      "source": "E_6037_4",
      "target": "20_5916_10",
      "weight": 9.1407
    },
    {
      "source": "16_12678_9",
      "target": "17_13508_9",
      "weight": 9.1285
    },
    {
      "source": "E_6037_4",
      "target": "7_691_10",
      "weight": 9.0795
    },
    {
      "source": "7_5741_1",
      "target": "9_2762_1",
      "weight": 9.0354
    },
    {
      "source": "20_15589_9",
      "target": "24_6044_10",
      "weight": 9.0061
    },
    {
      "source": "7_462_1",
      "target": "9_3056_1",
      "weight": 8.9466
    },
    {
      "source": "14_2268_9",
      "target": "24_6044_10",
      "weight": 8.9306
    },
    {
      "source": "7_5741_1",
      "target": "9_3056_1",
      "weight": 8.8529
    },
    {
      "source": "E_26865_9",
      "target": "17_7767_9",
      "weight": 8.5572
    },
    {
      "source": "7_462_1",
      "target": "9_2762_1",
      "weight": 8.4454
    },
    {
      "source": "10_3837_10",
      "target": "16_15223_10",
      "weight": -8.3009
    },
    {
      "source": "22_3064_10",
      "target": "25_4717_10",
      "weight": 8.2153
    },
    {
      "source": "14_2268_9",
      "target": "22_3064_10",
      "weight": 8.1619
    },
    {
      "source": "18_6101_10",
      "target": "19_1445_10",
      "weight": 7.7871
    },
    {
      "source": "4_13154_9",
      "target": "19_7477_9",
      "weight": 7.671
    },
    {
      "source": "7_6861_9",
      "target": "20_15589_10",
      "weight": 7.5745
    },
    {
      "source": "16_25_9",
      "target": "22_3064_10",
      "weight": 7.4528
    },
    {
      "source": "7_6861_9",
      "target": "19_7477_9",
      "weight": 7.3483
    },
    {
      "source": "14_2268_9",
      "target": "22_3551_10",
      "weight": 7.2459
    },
    {
      "source": "16_4298_10",
      "target": "21_5943_10",
      "weight": 7.1651
    },
    {
      "source": "7_6861_9",
      "target": "14_2268_9",
      "weight": 7.1201
    },
    {
      "source": "19_1445_10",
      "target": "25_4717_10",
      "weight": -7.0469
    },
    {
      "source": "E_26865_9",
      "target": "23_12237_10",
      "weight": 7.0348
    },
    {
      "source": "19_1445_10",
      "target": "20_15589_10",
      "weight": 6.8438
    },
    {
      "source": "E_2329_7",
      "target": "20_15589_10",
      "weight": 6.6555
    },
    {
      "source": "20_15589_10",
      "target": "23_12237_10",
      "weight": 6.5631
    },
    {
      "source": "E_26865_9",
      "target": "11_3544_9",
      "weight": 6.2932
    },
    {
      "source": "19_1445_10",
      "target": "22_3551_10",
      "weight": 6.2703
    },
    {
      "source": "24_13277_10",
      "target": "25_553_10",
      "weight": 6.2087
    },
    {
      "source": "E_6037_4",
      "target": "21_5943_10",
      "weight": 6.1776
    },
    {
      "source": "E_26865_9",
      "target": "7_3099_9",
      "weight": 6.1588
    },
    {
      "source": "4_13154_9",
      "target": "16_25_9",
      "weight": 6.1533
    },
    {
      "source": "17_7178_10",
      "target": "21_5943_10",
      "weight": 6.1236
    },
    {
      "source": "18_6101_10",
      "target": "20_5916_10",
      "weight": 6.0802
    },
    {
      "source": "16_4298_10",
      "target": "19_1445_10",
      "weight": 5.9427
    },
    {
      "source": "21_5943_10",
      "target": "25_9975_10",
      "weight": 5.8729
    },
    {
      "source": "E_6037_4",
      "target": "20_15589_10",
      "weight": 5.8141
    },
    {
      "source": "7_5741_1",
      "target": "9_13483_1",
      "weight": 5.8099
    },
    {
      "source": "4_13154_9",
      "target": "14_2268_9",
      "weight": 5.7326
    },
    {
      "source": "24_13277_10",
      "target": "25_4886_10",
      "weight": 5.6825
    },
    {
      "source": "4_13154_9",
      "target": "24_6044_10",
      "weight": 5.6335
    },
    {
      "source": "20_15589_9",
      "target": "22_3064_10",
      "weight": 5.5873
    },
    {
      "source": "19_1445_10",
      "target": "23_12237_10",
      "weight": 5.3985
    },
    {
      "source": "20_15589_10",
      "target": "27_22605_10",
      "weight": 5.3126
    },
    {
      "source": "E_6037_4",
      "target": "20_15589_9",
      "weight": 5.253
    },
    {
      "source": "19_1445_10",
      "target": "22_11220_10",
      "weight": 5.1107
    },
    {
      "source": "21_5943_10",
      "target": "22_3551_10",
      "weight": 5.0867
    },
    {
      "source": "22_3064_10",
      "target": "25_9975_10",
      "weight": -4.8781
    },
    {
      "source": "16_12678_9",
      "target": "20_15589_9",
      "weight": 4.8432
    },
    {
      "source": "15_4494_4",
      "target": "18_6101_10",
      "weight": 4.8069
    },
    {
      "source": "20_5916_10",
      "target": "22_11220_10",
      "weight": 4.7577
    },
    {
      "source": "19_7477_9",
      "target": "22_3064_10",
      "weight": 4.7181
    },
    {
      "source": "E_26865_9",
      "target": "27_22605_10",
      "weight": 4.7135
    },
    {
      "source": "E_6037_4",
      "target": "24_13277_10",
      "weight": 4.6682
    },
    {
      "source": "4_13154_9",
      "target": "22_3064_10",
      "weight": 4.6172
    },
    {
      "source": "21_5943_10",
      "target": "22_3064_10",
      "weight": 4.5972
    },
    {
      "source": "E_2329_7",
      "target": "10_3837_10",
      "weight": -4.5763
    },
    {
      "source": "15_4494_4",
      "target": "17_7178_10",
      "weight": 4.5562
    },
    {
      "source": "16_25_9",
      "target": "22_3551_10",
      "weight": 4.5388
    },
    {
      "source": "20_15589_9",
      "target": "22_3551_10",
      "weight": 4.5118
    },
    {
      "source": "E_26865_9",
      "target": "24_15627_10",
      "weight": 4.4528
    },
    {
      "source": "7_3099_9",
      "target": "17_7767_9",
      "weight": -4.4443
    },
    {
      "source": "24_6044_10",
      "target": "27_22605_10",
      "weight": -4.4125
    },
    {
      "source": "6_4012_7",
      "target": "18_8959_10",
      "weight": 4.3859
    },
    {
      "source": "E_2329_7",
      "target": "20_15589_9",
      "weight": 4.3709
    },
    {
      "source": "E_2329_7",
      "target": "19_7477_9",
      "weight": 4.3019
    },
    {
      "source": "7_462_1",
      "target": "9_13483_1",
      "weight": 4.291
    },
    {
      "source": "24_5668_10",
      "target": "25_4717_10",
      "weight": -4.2839
    },
    {
      "source": "20_5916_10",
      "target": "21_5943_10",
      "weight": 4.2612
    },
    {
      "source": "21_5943_10",
      "target": "24_5999_10",
      "weight": 4.2276
    },
    {
      "source": "7_6861_9",
      "target": "24_6044_10",
      "weight": 4.1732
    },
    {
      "source": "E_6037_4",
      "target": "18_8959_10",
      "weight": 4.1174
    },
    {
      "source": "E_2329_7",
      "target": "18_6101_10",
      "weight": 3.9835
    },
    {
      "source": "11_3544_9",
      "target": "17_7767_9",
      "weight": -3.9787
    },
    {
      "source": "20_15589_10",
      "target": "24_15627_10",
      "weight": 3.9773
    },
    {
      "source": "E_2329_7",
      "target": "20_5916_10",
      "weight": 3.9757
    },
    {
      "source": "15_4494_4",
      "target": "25_3264_10",
      "weight": 3.9347
    },
    {
      "source": "18_6101_10",
      "target": "22_11220_10",
      "weight": 3.9311
    },
    {
      "source": "21_5943_10",
      "target": "24_13277_10",
      "weight": 3.9042
    },
    {
      "source": "E_6037_4",
      "target": "22_11220_10",
      "weight": 3.8934
    },
    {
      "source": "21_5943_10",
      "target": "25_4886_10",
      "weight": 3.7747
    },
    {
      "source": "20_5916_10",
      "target": "22_3551_10",
      "weight": 3.7646
    },
    {
      "source": "7_462_1",
      "target": "10_14174_1",
      "weight": 3.7636
    },
    {
      "source": "18_8959_10",
      "target": "21_5943_10",
      "weight": 3.7616
    },
    {
      "source": "7_5741_1",
      "target": "10_14174_1",
      "weight": 3.7222
    },
    {
      "source": "E_6037_4",
      "target": "18_6101_10",
      "weight": 3.7197
    },
    {
      "source": "E_2329_7",
      "target": "17_7178_10",
      "weight": 3.7055
    },
    {
      "source": "6_4012_7",
      "target": "17_13250_7",
      "weight": 3.7022
    },
    {
      "source": "20_5916_10",
      "target": "25_4717_10",
      "weight": 3.6425
    },
    {
      "source": "16_15223_10",
      "target": "25_9975_10",
      "weight": 3.5917
    },
    {
      "source": "16_12678_9",
      "target": "17_7767_9",
      "weight": -3.5559
    },
    {
      "source": "19_1445_10",
      "target": "25_9975_10",
      "weight": 3.5303
    },
    {
      "source": "4_13154_9",
      "target": "8_3099_9",
      "weight": -3.5182
    },
    {
      "source": "10_3837_10",
      "target": "25_9975_10",
      "weight": 3.5113
    },
    {
      "source": "E_6037_4",
      "target": "24_5999_10",
      "weight": -3.5074
    },
    {
      "source": "24_5999_10",
      "target": "25_9975_10",
      "weight": -3.4839
    },
    {
      "source": "0_13727_7",
      "target": "17_13250_7",
      "weight": 3.4628
    },
    {
      "source": "15_4494_4",
      "target": "16_4298_10",
      "weight": 3.4196
    },
    {
      "source": "8_3099_9",
      "target": "17_7767_9",
      "weight": -3.3883
    },
    {
      "source": "17_7178_10",
      "target": "25_3264_10",
      "weight": 3.3728
    },
    {
      "source": "E_6037_4",
      "target": "10_3837_10",
      "weight": -3.359
    },
    {
      "source": "22_3064_10",
      "target": "24_6044_10",
      "weight": -3.2945
    },
    {
      "source": "E_2329_7",
      "target": "21_5943_10",
      "weight": 3.2941
    },
    {
      "source": "20_5916_10",
      "target": "25_9975_10",
      "weight": 3.2354
    },
    {
      "source": "21_5943_10",
      "target": "25_3264_10",
      "weight": 3.232
    },
    {
      "source": "22_3064_10",
      "target": "24_13277_10",
      "weight": 3.2295
    },
    {
      "source": "16_4298_10",
      "target": "17_7178_10",
      "weight": 3.2275
    },
    {
      "source": "19_1445_10",
      "target": "25_4886_10",
      "weight": 3.2249
    },
    {
      "source": "4_7671_4",
      "target": "6_4662_4",
      "weight": 3.187
    },
    {
      "source": "21_5943_10",
      "target": "27_22605_10",
      "weight": 3.1704
    },
    {
      "source": "6_4662_4",
      "target": "15_4494_4",
      "weight": 3.1641
    },
    {
      "source": "7_6861_9",
      "target": "22_3551_10",
      "weight": 3.1531
    }
  ]
}
```

## Your analysis tasks

1. **Semantic clusters** — Identify the main semantic clusters and what concept each represents.
2. **Computational pathway** — Trace the main activation path from input tokens to the output token.
3. **High-influence & bottleneck nodes** — Which nodes/supernodes are most critical? Are there bottlenecks?
4. **Circuit function** — Is this factual recall, arithmetic reasoning, pattern matching, or something else?
5. **Reasoning vs. memorization** — Does this circuit reflect robust algorithmic reasoning or surface-level memorization? Cite specific nodes.
6. **Generalization analysis** — Which circuit components are task-general vs. task-specific to the default task? Would those components transfer to the counterfactual task?

## Final verdict (REQUIRED — answer this last, after your analysis)

Based solely on what this circuit reveals about the model's internal computation, predict whether the test model will correctly answer instances of the counterfactual task.

End your response with exactly this format:
```
GENERALIZATION VERDICT: [YES or NO]
CONFIDENCE: [integer 0-100]%
KEY EVIDENCE: [one sentence citing the specific circuit component(s) that drive this prediction]
```

```

</details>

### LLM Analysis

1. **Semantic clusters**

Main clusters are quite clear from the supernodes and pinned nodes:

- **`capital` supernode**: nodes like `15_4494_4`, `3_13984_4`, `1_1000_4`, plus downstream `16_4298_10`, `7_691_10`, `20_5916_10`, `25_3264_10`. This cluster represents the prompt asking for a **capital city** / completion in the semantic class “capital”.
- **`state` supernode**: `0_13727_7`, `6_4012_7`, and also unpinned `17_13250_7`. This is the **state/governmental unit** part of the relation.
- **`Texas` supernode**: `4_13154_9`, `14_2268_9`, `16_25_9`, `20_15589_9`, `20_15589_10`, with some Dallas-coupled nodes like `19_7477_9` and `22_3064_10`. This cluster is the dominant **entity-specific retrieval of Texas from Dallas**.
- **`preposition followed by place name` / location framing**: `19_1445_10`, `18_6101_10`, `24_6044_10`. These look like generic **place-name / locative completion scaffolding**.
- **`capital cities / say a capital city`**: `21_5943_10`, `17_7178_10`, `18_6101_10`. This is a generic **output-space restriction toward city/capital names**.

So the circuit is organized around:
1. detect the query type (“capital of the state…”),
2. identify the relevant state from Dallas,
3. constrain the answer to a capital-city token,
4. push specifically toward **Austin**.

---

2. **Computational pathway**

The main path is:

- **Input token embeddings**
  - `E_26865_9` = embedding for `" Dallas"`
  - `E_2329_7` = embedding for `" state"`
  - `E_6037_4` = embedding for `" capital"`

These seed three partially separate streams.

### A. Dallas → Texas stream
This is the strongest task-specific path.

- `E_26865_9 -> 19_7477_9` (“Dallas”), weight 68.85
- `E_26865_9 -> 20_15589_9` (“Texas”), weight 62.20
- `E_26865_9 -> 20_15589_10` (“Texas”), weight 43.26
- also to earlier Texas nodes: `4_13154_9`, `14_2268_9`, `16_25_9`, `7_6861_9`

This shows that the model is not visibly computing “find the state containing Dallas” via an explicit relational procedure. Instead, the Dallas token directly activates a **Texas concept bundle** across many layers.

Then Texas propagates forward:

- `14_2268_9 -> 16_25_9 -> 20_15589_9`
- `18_8959_10 -> 20_15589_10`
- `20_15589_10 -> 27_22605_10` (“ Austin”), weight 5.31

So a late Texas representation directly contributes to Austin logits.

### B. Capital query framing
- `E_6037_4 -> 15_4494_4` (“capital”), weight 49.32
- `E_6037_4 -> 1_1000_4`, `3_13984_4`, `6_4662_4`
- then into capital/city-output nodes:
  - `15_4494_4 -> 18_6101_10` (“capital cities”)
  - `15_4494_4 -> 17_7178_10` (“government buildings”)
  - `15_4494_4 -> 16_4298_10` (“capital”)

Then:
- `18_6101_10 -> 21_5943_10` (“cities”), weight 14.02
- `21_5943_10 -> 23_12237_10` (“Cities and states names”), weight 10.83
- `21_5943_10 -> 27_22605_10` (“ Austin”), weight 3.17

This looks like a generic answer-type pathway: the model expects a **city name, specifically a capital city**.

### C. State/government framing
- `E_2329_7 -> 0_13727_7` (“State”), weight 11.79
- `E_2329_7 -> 6_4012_7` (“state abbreviations”), weight 16.39
- `E_2329_7 -> 18_8959_10` (“government/state”), weight 19.63

Then:
- `6_4012_7 -> 18_8959_10`
- `18_8959_10 -> 20_15589_10` (“Texas”), weight 10.58
- `18_8959_10 -> 21_5943_10` (“cities”), weight 3.76

This is a generic government/state schema that supports the Dallas→Texas retrieval and capital-city completion.

### Overall pathway
The dominant computation is:

**“Dallas” -> Texas-specific features**  
plus  
**“capital” + “state” -> capital-city/state query framing**  
which converge on  
**Texas + capital-city output features -> logit for “ Austin”**

---

3. **High-influence & bottleneck nodes**

Most critical nodes/supernodes:

- **`0_13727_7` (“State”)**: influence 0.503, high for an early generic query-structure node.
- **`15_4494_4` (“capital”)**: influence 0.469, very strong answer-type cue.
- **`21_5943_10` (“cities”)** and **`23_12237_10` (“Cities and states names”)**: these are late-stage output-shaping nodes.
- **Texas bundle**:
  - `20_15589_9` influence 0.326
  - `20_15589_10` influence 0.258
  - `14_2268_9`, `16_25_9`, `4_13154_9`, `7_6861_9` all materially support Texas activation
- **`19_1445_10`** (“Downtowns of cities”), influence 0.383: despite noisy label, it appears to function as a place/city completion scaffold.

### Bottlenecks
Yes, there are two clear bottlenecks:

- **Texas bottleneck**: the circuit repeatedly compresses toward a specific Texas representation (`20_15589_9`, `20_15589_10`) from Dallas. This is the task-specific entity bottleneck.
- **Capital-city output bottleneck**: `18_6101_10 -> 21_5943_10 -> 23_12237_10` is a generic output bottleneck restricting the answer to a city/capital name.

The final Austin logit seems to depend on the conjunction of:
1. **Texas identity**, and
2. **capital-city answer framing**.

---

4. **Circuit function**

This is best characterized as **entity-linked factual/geographic recall with light compositional scaffolding**, not arithmetic or abstract reasoning.

It contains a two-step task at the prompt level:
- Dallas is in Texas
- capital of Texas is Austin

But internally, the extracted circuit mostly looks like:
- retrieve **Texas** from **Dallas**
- invoke a **capital city answer template**
- output **Austin**

So it is closer to **factual association over geographic entities** than a robust symbolic two-hop procedure.

---

5. **Reasoning vs. memorization**

This circuit looks more like **surface-level memorized association/compressed factual retrieval** than robust algorithmic reasoning.

Evidence:

- The strongest task-specific activations are **direct Dallas→Texas** links:
  - `E_26865_9 -> 20_15589_9` weight 62.20
  - `E_26865_9 -> 20_15589_10` weight 43.26
  - `E_26865_9 -> 4_13154_9`, `14_2268_9`, `16_25_9`, `7_6861_9`
- There is no explicit generic “city contained in state -> retrieve state -> then retrieve capital of that state” chain represented by abstract relation nodes. Instead, the middle of the circuit is dominated by **Texas-specific features**.
- Some Texas nodes are labeled oddly but still highly specific:
  - `16_25_9` = “Texas legal documents”
  - `7_6861_9` = “Texas legal cases”
  These indicate broad **Texas-associated lexical/statistical memory**, not clean relational operators.
- The capital side is generic, but the state-identification side is not. The model seems to know “this is a capital question” in a task-general way, but the answer depends on retrieving the memorized state/entity cluster.

So there is some composition, but it is weakly compositional:
- generic **capital-question framing**
- plus **memorized Dallas/Texas knowledge**
- plus likely memorized **Texas/Austin knowledge**

That is not strong evidence of robust reasoning over a rewritable world model.

---

6. **Generalization analysis**

### Task-general components
These likely would transfer to the counterfactual setting:

- **capital query framing**:
  - `E_6037_4`, `15_4494_4`, `3_13984_4`, `1_1000_4`
- **state/government schema**:
  - `E_2329_7`, `0_13727_7`, `6_4012_7`, `18_8959_10`
- **city/capital-name output formatting**:
  - `18_6101_10`, `21_5943_10`, `23_12237_10`, `19_1445_10`

These components encode “the answer should be a capital city associated with a state.”

### Task-specific components
These are much less likely to transfer:

- **Dallas/Texas retrieval**
  - `E_26865_9 -> 20_15589_9`, `20_15589_10`, `4_13154_9`, `14_2268_9`, `16_25_9`, `7_6861_9`
- **Texas-conditioned push to Austin**
  - `20_15589_10 -> 27_22605_10`
- coupled Texas/Dallas nodes:
  - `19_7477_9` (“Dallas”)
  - `22_3064_10` (“Texas/Dallas”)

These are anchored in the default-world geography.

### Would they transfer to the counterfactual task?
Based on this circuit, probably **not**.

Why:
- The circuit does not show a mechanism for loading or privileging a new temporary mapping like “In this world, Boise is the capital of Texas”.
- Instead, it shows a strong pre-existing path from **Dallas to Texas** and then a late Texas-to-Austin push.
- The generic capital/state machinery would still help answer in the form of a city name, but the **state→capital mapping appears hardwired to the memorized default world**, not dynamically rewritten from context in this circuit.

So even if the counterfactual preamble mentions a new capital of Texas, this circuit gives stronger evidence for fallback to **Austin** than for updating to **Boise**.

GENERALIZATION VERDICT: NO  
CONFIDENCE: 89%  
KEY EVIDENCE: The circuit is dominated by direct Dallas→Texas retrieval (`E_26865_9 -> 20_15589_9/20_15589_10`) and a Texas-conditioned push to the Austin logit (`20_15589_10 -> 27_22605_10`), with no clear mechanism for contextually rewriting the Texas→capital mapping.

---

_Report generated by `experiments/analyze_vignette.py`_