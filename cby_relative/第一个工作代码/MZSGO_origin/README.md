# MZSGO:multimodal zero-shot protein function annotation via evolutionary signals and textual semantics

<div align="center">
  <img width="800" alt="MZSGO Framework Overview" src="https://github.com/toxic-byte/MZSGO/blob/main/images/model.png">
</div>

## 📖 Overview

**MZSGO** is a novel multimodal framework designed for **Zero-Shot Protein Function Prediction**.

Traditional deep learning approaches often rely solely on sequence patterns and treat functional labels as mere categorical tags, failing to capture the rich semantic information embedded in their definitions. MZSGO bridges this gap by fusing evolutionary signals from **Protein Language Models (PLMs)** with semantic features derived from **Large Language Models (LLMs)**.

By employing an **Adaptive Gated Fusion** mechanism, MZSGO effectively aligns sequence-based and text-based modalities. This allows for robust predictions of Gene Ontology (GO) terms, including unseen long-tail labels and novel functional categories.

### Key Features
- **Multimodal Integration**: Combines protein sequence embeddings with domain-specific textual descriptions.
- **Zero-Shot Generalization**: Leverages LLMs to construct a semantic space, enabling prediction of GO labels not seen during training.
- **Adaptive Gated Fusion**: Dynamically balances the influence of sequence and textual modalities to reduce noise.
- **Robustness**: Outperforms state-of-the-art methods on standard benchmarks, particularly in zero-shot scenarios.

## 🛠️ Installation

```bash
git clone https://github.com/toxic-byte/MZSGO.git
cd MZSGO
pip install -r requirements.txt
```

## 📂 Data Preparation

To run the model, you need to download the pre-computed embeddings.

1. Download **`embeddings_cache.zip`** from [Google Drive](https://drive.google.com/drive/u/0/folders/1KAOMWGNiqVIhKJfaffhX5GP0B1ITsj4r).
2. Unzip the file and place the contents into the `data/embeddings_cache/` directory.

**Directory Structure:**
```text
MZSGO/
├── data/
│   └── embeddings_cache/
├── images/
├── utils/
├── main.py
├── predict.py
└── ...
```

## 🚀 Quick Start

### 1. Prediction

**Predict specific ontology (BP/MF/CC):**
```bash
python predict.py --fasta example.fasta --pred_mode mf 
```

**Predict using a specific list of GO terms:**
```bash
python predict.py --fasta example.fasta --go_terms go_terms.txt
```

**Custom Zero-Shot Prediction:**
```bash
python predict.py --fasta example.fasta \
    --custom_go "protein kinase activity" \
    --custom_ontology mf 
```

### 2. Training

**Train the model on a specific ontology:**

```bash
python main.py --run_mode full --onto bp --epoch_num 30
```

### 3. Evaluation

**Standard Evaluation:**
```bash
# Get results on the test set
python test.py --run_mode full
```

**Zero-Shot Analysis:**
```bash
python test.py --run_mode zero
```
