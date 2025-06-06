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

📌 *Coming soon – paper under submission.*  
If you use this resource, please ⭐ star the repo and stay tuned for citation info.

---


## 🙏 Acknowledgements 

The authors thank the funding from the Horizon Europe research and innovation programme under the Marie Skłodowska-Curie Grant Agreement No. 101073351. The authors also thank the financial support supplied by the Consellería de Cultura, Educación, Formación Profesional e Universidades (accreditation 2019-2022 ED431G/01, ED431B 2022/33) and the European Regional Development Fund, which acknowledges the CITIC Research Center in ICT of the University of A Coruña as a Research Center of the Galician University System and the project PID2022-137061OB-C21 (Ministerio de Ciencia e Innovación, Agencia Estatal de Investigación, Proyectos de Generación de Conocimiento; supported by the European Regional Development Fund). The authors also thank the funding of project PLEC2021-007662 (MCIN/AEI/10.13039/501100011033, Ministerio de Ciencia e Innovación, Agencia Estatal de Investigación, Plan de Recuperación, Transformación y Resiliencia, Unión Europea-Next Generation EU).

---

## 📬 Contact

For questions, please reach out via email: `michelejoshua.maggini@usc.es` or `paloma.piot@udc.es`




