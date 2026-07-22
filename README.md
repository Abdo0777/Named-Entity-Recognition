# Named Entity Recognition System

An end-to-end NLP pipeline that identifies and classifies named entities — people, organizations, locations, and miscellaneous proper nouns — in free text. Built and compared across four architectures, from scratch and fine-tuned, and deployed as a live interactive web app.

## Overview

This project implements and compares:
- **LSTM** — unidirectional, built from scratch
- **BiLSTM** — bidirectional, built from scratch
- **BiLSTM + CRF** — bidirectional with a Conditional Random Field layer for structured decoding
- **DistilBERT** (fine-tuned) — pretrained transformer via HuggingFace

All models are trained on the **CoNLL-2003** dataset using the IOB tagging scheme, and evaluated with entity-level precision, recall, and F1 for four categories: `PER` (person), `ORG` (organization), `LOC` (location), and `MISC` (miscellaneous).

## Results (test set)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| LSTM | 0.71 | 0.64 | 0.67 |
| BiLSTM | 0.78 | 0.73 | 0.75 |
| BiLSTM + CRF | 0.82 | 0.74 | 0.78 |
| **DistilBERT (fine-tuned)** | **0.88** | **0.89** | **0.88** |

Fine-tuning a pretrained transformer gives the largest single jump in performance, since it already encodes language structure before ever seeing CoNLL-2003 data. The CRF layer improves boundary detection over plain BiLSTM by enforcing valid label transitions (e.g. blocking an `I-PER` tag from directly following a `B-ORG` tag).


## Project structure

ner_deployment/
├── app.py # Gradio web app
├── requirements.txt # Python dependencies
├── distilbert-ner/ # Fine-tuned model + tokenizer (not included — see below)
├── notebook.ipynb # Full training pipeline (all 4 architectures)
└── README.md

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/Abdo0777/Named-Entity-Recognition
cd Named-Entity-Recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the trained model

The fine-tuned DistilBERT weights aren't stored in this repo (model files are too large for plain Git). Download them from **[add your link here — Google Drive / Kaggle output / Hugging Face Hub]** and place the folder here so it looks like:

distilbert-ner/
├── config.json
├── model.safetensors
├── tokenizer.json
├── vocab.txt
└── id2label.json

### 4. Run the app

```bash
python app.py
```

Gradio will print a local URL, typically:
Running on local URL: http://127.0.0.1:7860

Open that link in your browser. Type any sentence and watch entities highlight live, or click one of the built-in examples.

## Retraining from scratch

The full pipeline — dataset loading, IOB tagging, GloVe embeddings, vocabulary building, all four model architectures, and evaluation — is in `named-entity-recognition.ipynb`. It's designed to run on **Kaggle with a GPU (T4)** end to end:

1. Enable GPU: Settings → Accelerator → GPU T4
2. Enable internet: Settings → Internet → On
3. Run all cells top to bottom
4. Trained models and the DistilBERT checkpoint are saved to `/kaggle/working/`

## Dataset

[CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003), loaded via HuggingFace Datasets.

## Tech stack

- PyTorch (LSTM / BiLSTM / BiLSTM+CRF)
- HuggingFace Transformers (DistilBERT fine-tuning)
- HuggingFace Datasets (CoNLL-2003)
- Gradio (deployment)
- GloVe (pretrained word embeddings)
