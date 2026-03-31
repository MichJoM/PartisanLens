# 🕵️‍♂️ PartisanLens: A Multilingual Dataset of Hyperpartisan and Conspiratorial Immigration Narratives in European Media

**PartisanLens** is a dataset focused on hyperpartisanship, stance detection, and PRCT, featuring human-authored rationales and detailed annotations.

---

## 📁 Repository Structure

```
partisanlens/
│
├── data/ 📦 Dataset, keywords & rationales
├── data_curation/ 🧪 Data sampling, statistics, and analysis scripts
│ ├── analysis/ 📊 Data analysis scripts
│ └── DPP_extraction.py
├── experiments/ 🧠 Model training, inference, rationale generation
│ ├── build-templated-rationales.py
│ ├── rephrase-rationales.py
│ ├── inference.py
│ └── finetune.py
└── annotation_guidelines.pdf 📄 Annotation schema and instructions
```

---

## 📌 Dataset Overview

**PartisanLens** includes:

- 🔴🔵 **Hyperpartisan annotations** – identifying overtly partisan language  
- 🧭 **Stance detection** – determining whether the speaker is *pro*, *against*, or *neutral* towards immigration  
- 🧠 **PRCT labels** – Population Replacement Conspiracy Theories  

Each sample contains:
- A political *text segment*
- Task-specific labels (hyperpartisan, stance, PRCT)
- Span annotation (loaded language, name calling and appeal to fear)

---

## 🔬 Experiments

We provide Python scripts to explore how LLMs and finetuned models handle reasoning with rationales.

| Module | Description                                                                                  |
|--------|----------------------------------------------------------------------------------------------|
| 🧱 `build-templated-rationales.py` | Automatically build templated rationales from the span annotation                            |
| ✍️ `rephrase-rationales.py` | Rephrase or augment rationales using LLMs for more fluente and natural language explanations |
| 🤖 `inference.py` | Perform zero-shot or few-shot inference using LLMs                                           |
| 🎯 `finetune.py` | Finetune models with (or without) rationale supervision                                      |

### ✍️ Rephrasing Rationales — `rephrase-rationales.py`

This script uses a LLM to **rephrase and enrich templated rationales** for each instance in the dataset, while preserving the original task labels. The output is a step-by-step explanation in JSON format for each example.

#### 🔧 How to Run

```bash
python3 experiments/rephrase-rationales.py \
  --dataset data/train_templated_rationales.csv \
  --output data/train_rephrased_rationales.csv \
  --hf_token your_huggingface_token
```
#### 🔧 Arguments

| Argument                        | Type   | Required | Description                                                                                                                                                                           |
|---------------------------------|--------|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dataset`                     | `str`  | ✅ Yes    | Path to the input dataset (`.csv` or `.tsv`). Must include columns like `id`, `text`, `templated_rationales`, `hyperpartisan_gold_label`, `prct_gold_label`, and `stance_gold_label`. |
| `--output`                      | `str`  | ❌ No     | Path to the output file (`.csv`). Default: `rephrased-rationales.csv`.                                                                                                                |
| `--hf_token`                    | `str`  | ❌ No     | Hugging Face token (used to access gated models from the `unsloth` hub).                                                                                                              |


### 🤖 Inference — `inference.py`

This script performs **LLM-based inference** using zero-shot or few-shot prompting, either to generate **rationales** and **predict labels** or only **predict labels**. You can select different models and modes depending on your use case.

#### ▶️ How to Run

```bash
python3 experiments/inference.py \
  --dataset data/test.csv \
  --model llama3.3-70 \
  --mode rationales \
  --output data/predictions.tsv \
  --hf_token your_huggingface_token
```

#### 🧩 Modes of Operation

You can choose between two modes when running the script:

| Mode        | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `rationales`| 🔍 Generates natural language rationales (chain-of-thought explanations) for each input sentence. |
| `labels`    | 🏷️ Directly predicts the classification labels: `hyperpartisan`, `PRCT`, and `stance` — without generating a rationale. |

#### 🔧 Arguments

| Argument       | Type   | Required | Description                                                                 |
|----------------|--------|----------|-----------------------------------------------------------------------------|
| `--dataset`    | `str`  | ✅ Yes    | Path to the input dataset (`.csv` or `.tsv`). Must include a `text` column. |
| `--model`      | `str`  | ✅ Yes    | Model identifier. Must be one of: `llama3.1-8b`, `llama3.3-70`, `nemo`.     |
| `--output`     | `str`  | ❌ No     | Path to the output predictions file. Default: `rephrased-rationales.csv`.   |
| `--mode`       | `str`  | ❌ No     | Whether to generate `"rationales"` or `"labels"`. Default: `rationales`.    |
| `--hf_token`   | `str`  | ❌ No     | Hugging Face token for accessing gated models (e.g., LLaMA-3).              |

### 🚀 Fine-tuning — `finetune.py`

Fine-tune a model on the dataset with options for generating either rationales or labels.

```bash
python3 finetune.py \
  --dataset data/train.csv \
  --model MODEL_NAME llama3.3-70
```
#### 🔧 Arguments

| Argument           | Type   | Required | Description                                                                                                      |
|--------------------|--------|----------|------------------------------------------------------------------------------------------------------------------|
| `--dataset`        | `str`  | ✅ Yes    | Path to the input dataset (`.csv` or `.tsv`) containing the training data. Must include `text`and label columns. |
| `--model`          | `str`  | ✅ Yes    | Model to fine-tune. Must be one of: `llama3.1-8b`, `llama3.3-70`, `nemo`.                                        |
| `--new_model_name` | `str`  | ❌ No     | File name/path for saving the fine-tuned model and tokenizer. Default: `new-model`.                              |
| `--mode`           | `str`  | ❌ No     | Mode of fine-tuning: `"rationales"` for explanations or `"labels"` for only classification labels.               |
| `--hf_token`       | `str`  | ❌ No     | Hugging Face token for accessing gated models (e.g., LLaMA-3).                                                   |


---

## 📊 Data Curation

The `data_curation/` directory contains:

- 📈 Scripts for analyzing dataset composition  
- ⚖️ Sampling strategies used the create the dataset  
- 🧮 Statistical reports and visualizations  

---

## 📚 Annotation Guidelines

Full documentation of tasks, labeling protocols, and rationale-writing instructions are provided in:

📄 `annotation_guidelines.pdf`


---

## 💡 Use Cases

- 🧠 **Interpretability research using rationales**  
  Use the human-curated / LLM-improved rationales to evaluate and improve model transparency and explainability.

- 🔍 **Political bias and stance analysis**  
  Study how models detect hyperpartisan language and take stances toward immigration claims.

- 🤖 **Fine-tuning models with explanation supervision**  
  Train models not only to classify but also to generate or use rationales, improving generalization and trustworthiness.

---

## 📝 Citation

📌 @inproceedings{maggini-etal-2026-partisanlens,
    title = "{P}artisan{L}ens: A Multilingual Dataset of Hyperpartisan and Conspiratorial Immigration Narratives in {E}uropean Media",
    author = "Maggini, Michele Joshua  and
      Piot, Paloma  and
      P{\'e}rez, Anxo  and
      Marino, Erik Bran  and
      Montesinos, L{\'u}a Santamar{\'i}a  and
      Cotovio, Ana Lisboa  and
      Abu{\'i}n, Marta V{\'a}zquez  and
      Parapar, Javier  and
      Gamallo, Pablo",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-long.53/",
    doi = "10.18653/v1/2026.eacl-long.53",
    pages = "1171--1186",
    ISBN = "979-8-89176-380-7",
    abstract = "Detecting hyperpartisan narratives and Population Replacement Conspiracy Theories (PRCT) is essential to addressing the spread of misinformation. These complex narratives pose a significant threat, as hyperpartisanship drives political polarisation and institutional distrust, while PRCTs directly motivate real-world extremist violence, making their identification critical for social cohesion and public safety. However, existing resources are scarce, predominantly English-centric, and often analyse hyperpartisanship, stance, and rhetorical bias in isolation rather than as interrelated aspects of political discourse. To bridge this gap, we introduce PartisanLens, the first multilingual dataset of 1617 hyperpartisan news headlines in Spanish, Italian, and Portuguese, annotated in multiple political discourse aspects. We first evaluate the classification performance of widely used Large Language Models (LLMs) on this dataset, establishing robust baselines for the classification of hyperpartisan and PRCT narratives. In addition, we assess the viability of using LLMs as automatic annotators for this task, analysing their ability to approximate human annotation. Results highlight both their potential and current limitations. Next, moving beyond standard judgments, we explore whether LLMs can emulate human annotation patterns by conditioning them on socio-economic and ideological profiles that simulate annotator perspectives. At last, we provide our resources and evaluation; PartisanLens supports future research on detecting partisan and conspiratorial narratives in European contexts."
}  
If you use this resource, please ⭐ star the repo and stay tuned for citation info.

---








