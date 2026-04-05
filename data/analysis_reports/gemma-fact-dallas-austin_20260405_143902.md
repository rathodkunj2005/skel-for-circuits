# Circuit Analysis Report

**Generated:** 2026-04-05 14:39:02  
**Vignette:** `data/outputs/Gemma/gemma-fact-dallas-austin_skeleton.json`  
**Model (scan):** `gemma-2-2b`  
**Analysis LLM:** `gpt-5.4`  
**Slug:** `gemma-fact-dallas-austin`  
**Elapsed:** 23.8s  

---

## Circuit Prompt

```
<bos>Fact: The capital of the state containing Dallas is
```

## Task Description

geographic factual recall

## Counterfactual Task

same question but with a less common US city instead of Dallas

---

## Full Analysis Prompt Sent to LLM

<details>
<summary>Click to expand full prompt</summary>

```
You are given a sparse feature circuit extracted from a language model in JSON format.

## Task context
- **Test model (whose circuit this is):** gemma-2-2b
- **Input prompt the circuit was extracted from:** `<bos>Fact: The capital of the state containing Dallas is`
- **Default task description:** geographic factual recall
- **Counterfactual task to predict generalization for:** same question but with a less common US city instead of Dallas

## Circuit summary
The circuit has been skeletonized into supernodes (semantic clusters of features).
Supernodes (each is a semantically coherent group of nodes):
  - **Output " Austin" (p=0.450)** → nodes: 27_22605_10
  - **Texas Legal Information** → nodes: 14_2268_9, 16_25_9
  - **Geographic Place Names** → nodes: 20_15589_10, 21_5943_10, 23_12237_10, E_26865_9
  - **in** → nodes: 24_6044_10
  - **Punctuation Markers** → nodes: E_2_0, E_18143_1, E_235292_2, E_714_3
  - **Sentence Structure Elements** → nodes: E_6037_4, E_576_5, E_2329_7, E_10751_8, E_603_10, 27_780_10
  - **Output " the" (p=0.055)** → nodes: 27_573_10

Key pinned nodes (highest-influence nodes selected by the skeletonization pipeline):
  - `27_22605_10` (layer 27) | influence=0.000 activation=0.000 | "Output " Austin" (p=0.450)"
  - `14_2268_9` (layer 14) | influence=0.000 activation=0.000 | "Texas"
  - `16_25_9` (layer 16) | influence=0.000 activation=0.000 | "Texas legal documents"
  - `20_15589_10` (layer 20) | influence=0.000 activation=0.000 | "Texas"
  - `21_5943_10` (layer 21) | influence=0.000 activation=0.000 | "cities"
  - `23_12237_10` (layer 23) | influence=0.000 activation=0.000 | "Cities and states names"
  - `E_26865_9` (layer E) | influence=0.000 activation=0.000 | "Emb: "  Dallas""
  - `24_6044_10` (layer 24) | influence=0.000 activation=0.000 | "in"
  - `E_2_0` (layer E) | influence=0.000 activation=0.000 | "Emb: " <bos>""
  - `E_18143_1` (layer E) | influence=0.000 activation=0.000 | "Emb: " Fact""
  - `E_235292_2` (layer E) | influence=0.000 activation=0.000 | "Emb: " :""
  - `E_714_3` (layer E) | influence=0.000 activation=0.000 | "Emb: "  The""
  - `E_6037_4` (layer E) | influence=0.000 activation=0.000 | "Emb: "  capital""
  - `E_576_5` (layer E) | influence=0.000 activation=0.000 | "Emb: "  of""
  - `E_2329_7` (layer E) | influence=0.000 activation=0.000 | "Emb: "  state""

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
      "node_id": "27_22605_10",
      "layer": "27",
      "feature_type": "logit",
      "clerp": "Output \" Austin\" (p=0.450)",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "14_2268_9",
      "layer": "14",
      "feature_type": "cross layer transcoder",
      "clerp": "Texas",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "16_25_9",
      "layer": "16",
      "feature_type": "cross layer transcoder",
      "clerp": "Texas legal documents",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "20_15589_10",
      "layer": "20",
      "feature_type": "cross layer transcoder",
      "clerp": "Texas",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "21_5943_10",
      "layer": "21",
      "feature_type": "cross layer transcoder",
      "clerp": "cities",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "23_12237_10",
      "layer": "23",
      "feature_type": "cross layer transcoder",
      "clerp": "Cities and states names",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "E_26865_9",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \"  Dallas\"",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "24_6044_10",
      "layer": "24",
      "feature_type": "cross layer transcoder",
      "clerp": "in",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "E_2_0",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \" <bos>\"",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "E_18143_1",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \" Fact\"",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "E_235292_2",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \" :\"",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "E_714_3",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \"  The\"",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "E_6037_4",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \"  capital\"",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "E_576_5",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \"  of\"",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "E_2329_7",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \"  state\"",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "E_10751_8",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \"  containing\"",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "E_603_10",
      "layer": "E",
      "feature_type": "embedding",
      "clerp": "Emb: \"  is\"",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "27_780_10",
      "layer": "27",
      "feature_type": "logit",
      "clerp": "Output \" not\" (p=0.073)",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "27_573_10",
      "layer": "27",
      "feature_type": "logit",
      "clerp": "Output \" the\" (p=0.055)",
      "influence": 0,
      "activation": 0,
      "pinned": true
    },
    {
      "node_id": "0_437_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Metadata and associative terms",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_478_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "code syntax",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_997_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "facts",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_1847_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "crystallization",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_1903_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "The",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_2115_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "%\\]",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_2616_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "already existing",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_2724_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "simplicity or ease",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_3461_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "experiences",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_4175_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "grand",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_4254_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "pretty",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_4823_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "technical/scientific text",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_4956_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "time",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_4993_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Stage or Period",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_5215_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "code",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_5369_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Names",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_5590_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Mobile phones, scripts",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_6078_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "What",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_6116_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "rare or formal language",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_6133_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "opinions and reviews",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_6421_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "work, projects, official activities",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_7344_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Fig",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_7376_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "writer",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_7465_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "aspects",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_7692_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "decrease",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_7812_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "raise",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_8008_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "she",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_8098_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "cost savings",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_8163_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "economics, politics, legal/science",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_8489_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "experiment findings",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_8540_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "unique",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_8642_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "actually",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_9026_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "(\\<",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_9637_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "Comp",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_9689_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "computer code and website design",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_10138_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "comments",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_10317_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": " single character variables and parts of words that look like names or list-like data structures",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_10345_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "picture",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_11577_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "trails or conservatism",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_11670_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "higher",
      "influence": 0,
      "activation": 0,
      "pinned": false
    },
    {
      "node_id": "0_11888_1",
      "layer": "0",
      "feature_type": "cross layer transcoder",
      "clerp": "straight",
      "influence": 0,
      "activation": 0,
      "pinned": false
    }
  ],
  "links": [
    {
      "source": "E_26865_9",
      "target": "20_15589_10",
      "weight": 43.259
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
      "source": "E_26865_9",
      "target": "24_6044_10",
      "weight": 24.8152
    },
    {
      "source": "14_2268_9",
      "target": "16_25_9",
      "weight": 20.2962
    },
    {
      "source": "E_2_0",
      "target": "0_10317_1",
      "weight": 18.6594
    },
    {
      "source": "14_2268_9",
      "target": "20_15589_10",
      "weight": 14.2462
    },
    {
      "source": "E_2_0",
      "target": "0_1903_1",
      "weight": 13.4896
    },
    {
      "source": "E_2_0",
      "target": "0_4823_1",
      "weight": 12.6827
    },
    {
      "source": "E_2_0",
      "target": "0_6421_1",
      "weight": 11.6902
    },
    {
      "source": "E_2_0",
      "target": "20_15589_10",
      "weight": -11.3914
    },
    {
      "source": "21_5943_10",
      "target": "23_12237_10",
      "weight": 10.8258
    },
    {
      "source": "16_25_9",
      "target": "20_15589_10",
      "weight": 10.5184
    },
    {
      "source": "E_2_0",
      "target": "0_7344_1",
      "weight": 10.4251
    },
    {
      "source": "E_18143_1",
      "target": "0_997_1",
      "weight": 10.004
    },
    {
      "source": "16_25_9",
      "target": "24_6044_10",
      "weight": 9.8284
    },
    {
      "source": "E_603_10",
      "target": "27_573_10",
      "weight": 9.5766
    },
    {
      "source": "14_2268_9",
      "target": "24_6044_10",
      "weight": 8.9306
    },
    {
      "source": "E_2_0",
      "target": "16_25_9",
      "weight": -8.7767
    },
    {
      "source": "E_603_10",
      "target": "27_780_10",
      "weight": 8.5379
    },
    {
      "source": "E_18143_1",
      "target": "0_7692_1",
      "weight": 8.1752
    },
    {
      "source": "E_18143_1",
      "target": "0_11577_1",
      "weight": 7.9087
    },
    {
      "source": "E_2_0",
      "target": "0_6133_1",
      "weight": 7.7839
    },
    {
      "source": "E_18143_1",
      "target": "0_4254_1",
      "weight": 7.4016
    },
    {
      "source": "E_18143_1",
      "target": "0_6078_1",
      "weight": 7.2275
    },
    {
      "source": "E_18143_1",
      "target": "0_5590_1",
      "weight": 7.2005
    },
    {
      "source": "E_18143_1",
      "target": "0_8642_1",
      "weight": 7.1615
    },
    {
      "source": "E_2_0",
      "target": "0_8489_1",
      "weight": 7.155
    },
    {
      "source": "E_26865_9",
      "target": "23_12237_10",
      "weight": 7.0348
    },
    {
      "source": "E_18143_1",
      "target": "0_9689_1",
      "weight": 6.9866
    },
    {
      "source": "E_18143_1",
      "target": "0_4175_1",
      "weight": 6.943
    },
    {
      "source": "E_18143_1",
      "target": "0_7376_1",
      "weight": 6.6659
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
      "source": "E_18143_1",
      "target": "0_11888_1",
      "weight": 6.5232
    },
    {
      "source": "E_2_0",
      "target": "0_478_1",
      "weight": 6.3989
    },
    {
      "source": "E_6037_4",
      "target": "21_5943_10",
      "weight": 6.1776
    },
    {
      "source": "E_18143_1",
      "target": "0_7465_1",
      "weight": 6.0666
    },
    {
      "source": "E_18143_1",
      "target": "0_10138_1",
      "weight": 5.9843
    },
    {
      "source": "E_18143_1",
      "target": "0_9637_1",
      "weight": 5.8299
    },
    {
      "source": "E_6037_4",
      "target": "20_15589_10",
      "weight": 5.8141
    },
    {
      "source": "E_18143_1",
      "target": "0_8540_1",
      "weight": 5.7275
    },
    {
      "source": "E_18143_1",
      "target": "0_2616_1",
      "weight": 5.4758
    },
    {
      "source": "20_15589_10",
      "target": "27_22605_10",
      "weight": 5.3126
    },
    {
      "source": "E_18143_1",
      "target": "0_7812_1",
      "weight": 5.2783
    },
    {
      "source": "E_18143_1",
      "target": "0_11670_1",
      "weight": 5.2751
    },
    {
      "source": "E_18143_1",
      "target": "0_8098_1",
      "weight": 5.2195
    },
    {
      "source": "E_18143_1",
      "target": "0_3461_1",
      "weight": 5.1661
    },
    {
      "source": "E_2_0",
      "target": "0_8008_1",
      "weight": 5.078
    },
    {
      "source": "E_18143_1",
      "target": "0_437_1",
      "weight": 5.071
    },
    {
      "source": "E_714_3",
      "target": "20_15589_10",
      "weight": 5.0519
    },
    {
      "source": "E_18143_1",
      "target": "0_2724_1",
      "weight": 4.9538
    },
    {
      "source": "E_26865_9",
      "target": "27_22605_10",
      "weight": 4.7135
    },
    {
      "source": "E_18143_1",
      "target": "0_4956_1",
      "weight": 4.6702
    },
    {
      "source": "E_18143_1",
      "target": "0_10345_1",
      "weight": 4.6072
    },
    {
      "source": "E_18143_1",
      "target": "16_25_9",
      "weight": 4.4622
    },
    {
      "source": "24_6044_10",
      "target": "27_22605_10",
      "weight": -4.4125
    },
    {
      "source": "E_18143_1",
      "target": "0_7344_1",
      "weight": 4.3386
    },
    {
      "source": "E_18143_1",
      "target": "0_10317_1",
      "weight": -4.3231
    },
    {
      "source": "E_18143_1",
      "target": "0_4993_1",
      "weight": 4.1351
    },
    {
      "source": "E_18143_1",
      "target": "24_6044_10",
      "weight": 3.9458
    },
    {
      "source": "E_18143_1",
      "target": "0_5215_1",
      "weight": 3.7132
    },
    {
      "source": "E_603_10",
      "target": "20_15589_10",
      "weight": 3.3421
    },
    {
      "source": "E_2_0",
      "target": "0_5369_1",
      "weight": 3.3067
    },
    {
      "source": "E_2329_7",
      "target": "21_5943_10",
      "weight": 3.2941
    },
    {
      "source": "21_5943_10",
      "target": "27_22605_10",
      "weight": 3.1704
    },
    {
      "source": "E_2_0",
      "target": "0_5215_1",
      "weight": 3.078
    },
    {
      "source": "23_12237_10",
      "target": "24_6044_10",
      "weight": 3.025
    },
    {
      "source": "16_25_9",
      "target": "23_12237_10",
      "weight": 3.0014
    },
    {
      "source": "E_18143_1",
      "target": "0_6116_1",
      "weight": 2.9289
    },
    {
      "source": "E_18143_1",
      "target": "0_6421_1",
      "weight": -2.8939
    },
    {
      "source": "E_2_0",
      "target": "23_12237_10",
      "weight": -2.8802
    },
    {
      "source": "23_12237_10",
      "target": "27_22605_10",
      "weight": 2.8116
    },
    {
      "source": "E_6037_4",
      "target": "23_12237_10",
      "weight": 2.6853
    },
    {
      "source": "E_2_0",
      "target": "27_573_10",
      "weight": 2.6214
    },
    {
      "source": "14_2268_9",
      "target": "23_12237_10",
      "weight": 2.6148
    },
    {
      "source": "E_2_0",
      "target": "0_6116_1",
      "weight": 2.5177
    },
    {
      "source": "E_576_5",
      "target": "16_25_9",
      "weight": 2.2294
    },
    {
      "source": "E_18143_1",
      "target": "0_9026_1",
      "weight": 2.123
    },
    {
      "source": "E_2329_7",
      "target": "23_12237_10",
      "weight": 2.07
    },
    {
      "source": "E_6037_4",
      "target": "24_6044_10",
      "weight": -2.0372
    },
    {
      "source": "E_2_0",
      "target": "0_11577_1",
      "weight": 1.9735
    },
    {
      "source": "E_2_0",
      "target": "0_9689_1",
      "weight": 1.8441
    },
    {
      "source": "E_603_10",
      "target": "27_22605_10",
      "weight": 1.8241
    },
    {
      "source": "E_2329_7",
      "target": "16_25_9",
      "weight": 1.7902
    },
    {
      "source": "E_714_3",
      "target": "24_6044_10",
      "weight": 1.6884
    },
    {
      "source": "E_18143_1",
      "target": "0_8489_1",
      "weight": 1.682
    },
    {
      "source": "E_18143_1",
      "target": "0_4823_1",
      "weight": 1.6532
    },
    {
      "source": "E_6037_4",
      "target": "16_25_9",
      "weight": 1.6213
    },
    {
      "source": "16_25_9",
      "target": "27_22605_10",
      "weight": 1.6138
    },
    {
      "source": "E_18143_1",
      "target": "23_12237_10",
      "weight": -1.6026
    },
    {
      "source": "E_235292_2",
      "target": "24_6044_10",
      "weight": 1.5693
    },
    {
      "source": "E_235292_2",
      "target": "20_15589_10",
      "weight": 1.5507
    },
    {
      "source": "E_235292_2",
      "target": "21_5943_10",
      "weight": 1.5503
    },
    {
      "source": "14_2268_9",
      "target": "27_22605_10",
      "weight": 1.5017
    },
    {
      "source": "E_2_0",
      "target": "0_4993_1",
      "weight": 1.4491
    },
    {
      "source": "E_2_0",
      "target": "0_8163_1",
      "weight": 1.423
    },
    {
      "source": "E_603_10",
      "target": "24_6044_10",
      "weight": -1.4173
    },
    {
      "source": "E_2_0",
      "target": "27_780_10",
      "weight": 1.411
    },
    {
      "source": "E_10751_8",
      "target": "16_25_9",
      "weight": -1.3813
    },
    {
      "source": "E_2_0",
      "target": "0_2115_1",
      "weight": 1.3545
    },
    {
      "source": "E_2_0",
      "target": "14_2268_9",
      "weight": -1.3321
    },
    {
      "source": "E_2_0",
      "target": "0_7376_1",
      "weight": 1.3175
    },
    {
      "source": "E_2_0",
      "target": "0_8098_1",
      "weight": 1.244
    },
    {
      "source": "E_2_0",
      "target": "0_1847_1",
      "weight": 1.202
    },
    {
      "source": "E_18143_1",
      "target": "20_15589_10",
      "weight": 1.1882
    },
    {
      "source": "E_2_0",
      "target": "0_4175_1",
      "weight": 1.1588
    },
    {
      "source": "E_2_0",
      "target": "0_10138_1",
      "weight": 1.1061
    },
    {
      "source": "E_18143_1",
      "target": "0_2115_1",
      "weight": 1.0762
    },
    {
      "source": "E_235292_2",
      "target": "23_12237_10",
      "weight": 1.0716
    },
    {
      "source": "E_2_0",
      "target": "0_4254_1",
      "weight": 1.0651
    },
    {
      "source": "E_10751_8",
      "target": "24_6044_10",
      "weight": -1.0514
    },
    {
      "source": "E_576_5",
      "target": "14_2268_9",
      "weight": 1.0396
    },
    {
      "source": "E_2_0",
      "target": "0_8642_1",
      "weight": 1.0311
    },
    {
      "source": "E_18143_1",
      "target": "0_6133_1",
      "weight": -1.0166
    },
    {
      "source": "E_2_0",
      "target": "0_997_1",
      "weight": 0.9597
    },
    {
      "source": "E_603_10",
      "target": "21_5943_10",
      "weight": 0.9399
    },
    {
      "source": "E_26865_9",
      "target": "27_573_10",
      "weight": -0.9305
    },
    {
      "source": "E_18143_1",
      "target": "0_1847_1",
      "weight": -0.9243
    },
    {
      "source": "E_235292_2",
      "target": "27_780_10",
      "weight": 0.9234
    },
    {
      "source": "E_2_0",
      "target": "0_11888_1",
      "weight": 0.8982
    },
    {
      "source": "E_10751_8",
      "target": "20_15589_10",
      "weight": -0.8834
    },
    {
      "source": "E_2_0",
      "target": "0_2724_1",
      "weight": 0.8584
    },
    {
      "source": "E_2_0",
      "target": "0_8540_1",
      "weight": 0.8578
    },
    {
      "source": "E_2_0",
      "target": "0_6078_1",
      "weight": 0.8416
    },
    {
      "source": "E_576_5",
      "target": "20_15589_10",
      "weight": 0.824
    },
    {
      "source": "E_714_3",
      "target": "16_25_9",
      "weight": 0.8119
    },
    {
      "source": "E_10751_8",
      "target": "21_5943_10",
      "weight": -0.797
    },
    {
      "source": "E_18143_1",
      "target": "0_5369_1",
      "weight": 0.7152
    },
    {
      "source": "21_5943_10",
      "target": "27_573_10",
      "weight": -0.6923
    },
    {
      "source": "E_714_3",
      "target": "23_12237_10",
      "weight": 0.6919
    },
    {
      "source": "E_2_0",
      "target": "0_437_1",
      "weight": 0.6897
    },
    {
      "source": "E_2_0",
      "target": "0_2616_1",
      "weight": 0.6843
    },
    {
      "source": "E_2_0",
      "target": "0_7812_1",
      "weight": 0.6771
    },
    {
      "source": "E_18143_1",
      "target": "0_8008_1",
      "weight": -0.6708
    },
    {
      "source": "E_2329_7",
      "target": "14_2268_9",
      "weight": 0.6613
    },
    {
      "source": "E_2_0",
      "target": "0_4956_1",
      "weight": 0.6583
    },
    {
      "source": "E_235292_2",
      "target": "27_573_10",
      "weight": 0.6582
    },
    {
      "source": "20_15589_10",
      "target": "21_5943_10",
      "weight": -0.6475
    },
    {
      "source": "E_18143_1",
      "target": "0_478_1",
      "weight": 0.6303
    },
    {
      "source": "E_2_0",
      "target": "0_10345_1",
      "weight": 0.6295
    },
    {
      "source": "E_2_0",
      "target": "0_3461_1",
      "weight": 0.6045
    },
    {
      "source": "E_18143_1",
      "target": "14_2268_9",
      "weight": 0.604
    },
    {
      "source": "E_576_5",
      "target": "21_5943_10",
      "weight": 0.5997
    },
    {
      "source": "E_10751_8",
      "target": "14_2268_9",
      "weight": 0.5847
    },
    {
      "source": "E_18143_1",
      "target": "21_5943_10",
      "weight": 0.5693
    },
    {
      "source": "E_2329_7",
      "target": "24_6044_10",
      "weight": 0.5222
    },
    {
      "source": "21_5943_10",
      "target": "24_6044_10",
      "weight": 0.5219
    },
    {
      "source": "E_2_0",
      "target": "27_22605_10",
      "weight": -0.5152
    },
    {
      "source": "E_235292_2",
      "target": "14_2268_9",
      "weight": 0.4967
    },
    {
      "source": "21_5943_10",
      "target": "27_780_10",
      "weight": -0.4962
    },
    {
      "source": "E_576_5",
      "target": "27_573_10",
      "weight": 0.4876
    },
    {
      "source": "E_576_5",
      "target": "27_780_10",
      "weight": 0.4811
    },
    {
      "source": "E_6037_4",
      "target": "27_22605_10",
      "weight": -0.4756
    },
    {
      "source": "E_10751_8",
      "target": "27_22605_10",
      "weight": -0.462
    },
    {
      "source": "E_714_3",
      "target": "14_2268_9",
      "weight": 0.4391
    },
    {
      "source": "E_2329_7",
      "target": "27_22605_10",
      "weight": 0.4285
    },
    {
      "source": "E_18143_1",
      "target": "27_573_10",
      "weight": -0.4021
    },
    {
      "source": "E_2_0",
      "target": "0_7692_1",
      "weight": 0.4017
    },
    {
      "source": "E_603_10",
      "target": "23_12237_10",
      "weight": 0.3979
    },
    {
      "source": "E_2_0",
      "target": "0_5590_1",
      "weight": 0.3977
    },
    {
      "source": "E_2_0",
      "target": "0_7465_1",
      "weight": 0.3906
    },
    {
      "source": "0_997_1",
      "target": "20_15589_10",
      "weight": 0.3761
    },
    {
      "source": "E_576_5",
      "target": "27_22605_10",
      "weight": -0.3678
    },
    {
      "source": "E_235292_2",
      "target": "16_25_9",
      "weight": 0.3592
    },
    {
      "source": "14_2268_9",
      "target": "21_5943_10",
      "weight": -0.321
    },
    {
      "source": "0_10317_1",
      "target": "20_15589_10",
      "weight": -0.3199
    },
    {
      "source": "E_18143_1",
      "target": "27_780_10",
      "weight": 0.3078
    },
    {
      "source": "E_10751_8",
      "target": "23_12237_10",
      "weight": 0.3072
    },
    {
      "source": "E_714_3",
      "target": "27_780_10",
      "weight": -0.3063
    },
    {
      "source": "0_9026_1",
      "target": "20_15589_10",
      "weight": 0.2976
    },
    {
      "source": "0_1903_1",
      "target": "20_15589_10",
      "weight": 0.2959
    },
    {
      "source": "E_2_0",
      "target": "0_11670_1",
      "weight": 0.2942
    },
    {
      "source": "E_235292_2",
      "target": "27_22605_10",
      "weight": 0.2883
    },
    {
      "source": "E_6037_4",
      "target": "14_2268_9",
      "weight": -0.2883
    },
    {
      "source": "E_26865_9",
      "target": "21_5943_10",
      "weight": 0.2671
    },
    {
      "source": "E_26865_9",
      "target": "27_780_10",
      "weight": -0.2573
    },
    {
      "source": "E_714_3",
      "target": "27_573_10",
      "weight": 0.2465
    },
    {
      "source": "E_10751_8",
      "target": "27_780_10",
      "weight": -0.2416
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

---

## LLM Analysis

1. **Semantic clusters**

- **Output " Austin"**: the final answer node `27_22605_10`, the dominant predicted completion.
- **Texas Legal Information**: `14_2268_9` ("Texas") and `16_25_9` ("Texas legal documents"). Despite the "legal documents" label, in-circuit these act as Texas-specific state identity features.
- **Geographic Place Names**: `E_26865_9` (" Dallas"), `20_15589_10` ("Texas"), `21_5943_10` ("cities"), `23_12237_10` ("Cities and states names"). This is the main geography/entity-recognition stack.
- **in**: `24_6044_10`, likely representing a containment/location relation.
- **Punctuation Markers / Sentence Structure Elements**: BOS, "Fact", ":", "The", "capital", "of", "state", "containing", "is". These provide prompt-format and syntactic scaffolding rather than substantive geographic content.
- **Output " the"**: a weaker competing continuation node `27_573_10`, likely reflecting generic continuation pressure.

2. **Computational pathway**

The main path is:

- **City token recognition**: `E_26865_9` strongly activates `20_15589_10` (weight 43.259), `16_25_9` (28.9526), and `14_2268_9` (26.9398). So the embedding for **Dallas** is directly mapped to **Texas**-specific latent features.
- **State consolidation**: `14_2268_9 -> 16_25_9` (20.2962), `14_2268_9 -> 20_15589_10` (14.2462), `16_25_9 -> 20_15589_10` (10.5184). Multiple Texas-related features reinforce one another.
- **Geographic typing / relation support**: `20_15589_10 -> 23_12237_10` (6.5631), `21_5943_10 -> 23_12237_10` (10.8258), and several prompt tokens activate `21_5943_10`/`23_12237_10`. This looks like a generic place-name schema around city/state entities.
- **Output readout**: strongest direct route is `20_15589_10 -> 27_22605_10` (5.3126), with weaker support from `23_12237_10` (2.8116), `21_5943_10` (3.1704), `16_25_9` (1.6138), and `14_2268_9` (1.5017). There is also a direct lexical shortcut `E_26865_9 -> 27_22605_10` (4.7135).

So the core computation is: **Dallas token -> Texas features -> Austin output**, with supporting generic geography structure.

3. **High-influence & bottleneck nodes**

Most critical nodes/supernodes:

- **`E_26865_9` (" Dallas")**: major upstream driver. It has the strongest outgoing links into the Texas pathway and even a direct link to the Austin logit.
- **`20_15589_10` ("Texas")**: likely the main bottleneck. It receives strong support from Dallas and other Texas features, then has the strongest positive edge to the Austin output.
- **Texas Legal Information supernode** (`14_2268_9`, `16_25_9`): supportive state-identification backbone.
- **`23_12237_10` ("Cities and states names")** and **`21_5943_10` ("cities")**: auxiliary geography schema, but less decisive than the Texas node.

Main bottleneck:
- **`20_15589_10`** is the clearest bottleneck because many upstream paths converge there, and it is the strongest single mediator into `" Austin"`.

Notably, `24_6044_10` ("in") is not a strong positive bottleneck for the answer; it actually has a **negative** edge to Austin (`24_6044_10 -> 27_22605_10 = -4.4125`). So the relation feature seems present but not driving the final factual completion.

4. **Circuit function**

This is primarily **geographic factual recall via entity-to-state-to-capital association**, not arithmetic or multi-step reasoning.

The circuit does not show a compositional "find state containing city, then retrieve capital" algorithm in a strong abstract sense. Instead it shows a heavily content-addressed retrieval pipeline centered on Dallas/Texas.

5. **Reasoning vs. memorization**

This circuit looks much closer to **surface-level memorized association** than robust algorithmic reasoning.

Evidence:
- The input city embedding `E_26865_9` directly drives **Texas** features with very large weights, especially `E_26865_9 -> 20_15589_10` (43.259), and also directly boosts the Austin output (`E_26865_9 -> 27_22605_10` = 4.7135).
- The strongest answer-producing node is **Texas-specific** (`20_15589_10`), not an abstract "state-containing-city" operator.
- The relation-like node `24_6044_10` ("in") does not positively mediate the answer; it suppresses Austin. That argues against a clean containment-reasoning chain.
- The generic geography nodes `21_5943_10` and `23_12237_10` exist, but they provide weaker support than the Dallas->Texas path.

So the model seems to recognize the prompt template and use a memorized mapping from **Dallas** to **Texas**, then from **Texas** to **Austin**, rather than executing a broadly reusable relational procedure.

6. **Generalization analysis**

**Task-general components**
- Prompt-format and syntax nodes: BOS, "Fact", "The capital of the state containing ... is".
- Generic geography schema: `21_5943_10` ("cities"), `23_12237_10` ("Cities and states names").
- Possibly some weak relation handling through `24_6044_10` ("in"), though it is not functioning as a positive answer mediator here.

These components should transfer to a prompt with another city.

**Task-specific components**
- `E_26865_9` (" Dallas")
- Texas-specific nodes `14_2268_9`, `16_25_9`, `20_15589_10`
- The output node `" Austin"` as the dominant answer

These are highly specific to the Dallas->Texas->Austin fact pattern.

**Would components transfer to the counterfactual?**
- The **template-recognition** part likely transfers.
- The **abstract geography** part may partially transfer.
- But the decisive pathway here is not abstract enough: it is anchored in a specific city embedding activating specific Texas features. For a less common city, success would require that the model possess an equally strong city->state feature mapping for that city. This circuit gives little evidence of a robust reusable mechanism that would compensate if such a memorized mapping is weak or absent.

So based on this circuit alone, I would expect degraded performance on less common cities.

```text
GENERALIZATION VERDICT: NO
CONFIDENCE: 81%
KEY EVIDENCE: The dominant path is the task-specific chain `E_26865_9` (" Dallas") -> `20_15589_10` ("Texas") -> `27_22605_10` (" Austin"), while the more abstract relation node `24_6044_10` ("in") is not the main positive mediator and even suppresses the Austin logit.
```

---

_Report generated by `experiments/analyze_vignette.py`_