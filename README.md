# Ontology-Aware Knowledge Distillation for Text2Cypher

An open-source research framework for distilling structured 4-step Ontology Chain-of-Thought (CoT) reasoning from Large Language Models into Edge-friendly Small Language Models (SLMs) for Graph Database (Neo4j) Query Generation.

---

## Overview

Translating natural language questions into Cypher graph queries (**Text-to-Cypher**) is essential for Knowledge Graph (KG) systems. Standard Direct Supervised Fine-Tuning (Direct SFT: `Question + Schema -> Cypher`) on small models (<3B parameters) frequently suffers from **structural graph hallucinations** (e.g., non-existent node labels, invalid relationship directions, or missing T-Box properties).

This repository implements **Ontology-Aware Knowledge Distillation (KD)**. Instead of raw query mapping, student models learn a structured 4-step reasoning process:
1. **`instance_extraction`**: Extracts A-Box entities, classes, and datatype properties.
2. **`relation_mapping`**: Maps entity connections to T-Box Object Properties with exact domain, range, and direction.
3. **`validation_check`**: Verifies schema constraints under the Closed World Assumption (CWA).
4. **`cypher`**: Synthesizes the verified executable Cypher query.

---

## Key Features

- **Ontology-Aware CoT Distillation**: Transfers structured 4-step graph reasoning from Teacher LLMs (Gemini / GPT-4o) to sub-3B SLMs (`Qwen2.5-1.5B-Instruct` / `Qwen2.5-3B-Instruct`).
- **3-Tier Hallucination Validator**: Automatically rejects candidate samples violating T-Box schema constraints prior to student training.
- **De-leakage Data Sanitation**: Audits the standard `neo4j/text2cypher-2024v1` benchmark dataset, eliminating 2,104 overlapping questions between train and test sets to yield a **37,450 clean training dataset**.
- **Edge Deployment Ready**: Supports merging LoRA adapters and exporting to **GGUF (Q4_K_M)** for real-time inference on edge CPUs via `llama.cpp` and Ollama.

---

## Repository Structure

```text
Text2Cypher/
├── configs/
│   └── config.py                 # Hyperparameters, API keys, paths, and database configs
├── data/
│   ├── raw/                      # Raw dataset downloads
│   └── processed/                # Cleaned train (37.4k) and test sets (4.8k)
├── src/                          # Core Engine Modules
│   ├── data/
│   │   ├── cleaner.py            # De-leakage data cleaning
│   │   └── formatter.py          # SFT dataset packaging (Baseline vs Proposed KD)
│   ├── extraction/
│   │   ├── teacher.py            # Multi-provider Teacher Extractor (Gemini / OpenAI / vLLM)
│   │   └── validator.py          # 3-Tier CWA Hallucination Validator
│   ├── prompts/
│   │   └── teacher_prompts.py    # Structured CoT prompt templates
│   ├── training/
│   │   └── trainer.py            # Student SLM SFT Trainer (Unsloth + QLoRA 4-bit)
│   ├── evaluation/
│   │   ├── metrics.py            # Google-BLEU & Execution Exact Match metrics
│   │   └── evaluator.py          # Benchmark evaluation pipeline
│   └── utils/
│       ├── logger.py             # Logging utilities
│       ├── json_parser.py        # Robust 4-step CoT JSON parser
│       └── cleanup.py            # Script maintenance utilities
├── scripts/                      # Sequential Execution Pipeline
│   ├── 01_prepare_data.py        # Stage 1: Data cleaning & de-leakage
│   ├── 02_distill_teacher.py     # Stage 2: Teacher distillation with 3-tier validation
│   ├── 03_export_sft_datasets.py # Stage 3: Export baseline & KD training sets
│   ├── 04_train_student.py       # Stage 4: Fine-tune student SLM models
│   ├── 05_evaluate.py            # Stage 5: Benchmark evaluation on test set
│   └── 06_demo_cli.py            # Stage 6: Interactive CLI Demo Playground
├── outputs/                      # Model adapters, GGUF exports, and benchmark reports
├── .gitignore                    # Git tracking rules
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## Quick Start

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/LionelHuanSi/Text2Cypher.git
cd Text2Cypher
pip install -r requirements.txt
```

### 2. Data Preparation (De-leakage Cleaning)
Download raw dataset and perform train/test question de-leakage:
```bash
python scripts/01_prepare_data.py
```

### 3. Knowledge Distillation
Extract 4-step CoT reasoning traces using Teacher LLM:
```bash
venv\scripts\activate
python scripts/02_distill_teacher.py
```

### 4. Format SFT Training Sets
Export baseline and proposed CoT KD datasets:
```bash
python scripts/03_export_sft_datasets.py
```

### 5. Student Fine-Tuning
Fine-tune student SLMs using QLoRA 4-bit SFT:
```bash
# Train Proposed Ontology CoT KD Model
python scripts/04_train_student.py --mode kd --model_id unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit

# Train Direct SFT Baseline Model
python scripts/04_train_student.py --mode baseline --model_id unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit
```

### 6. Benchmark Evaluation
Evaluate trained adapters on the test set:
```bash
python scripts/05_evaluate.py --adapter qwen2.5_1.5b_student_kd
```

### 7. Interactive CLI Playground
Launch interactive demo CLI:
```bash
python scripts/06_demo_cli.py
```

---

## Benchmark Evaluation Protocol

- **Google-BLEU**: Measures Lexical & Cypher syntax similarity against ground-truth queries across all test samples.
- **Execution Exact Match (%)**: Verifies actual database query results against standard Neo4j instances.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
