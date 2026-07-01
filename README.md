# Aspect-Based Sentiment Analysis (ABSA) — Telecom Customer Feedback

An end-to-end ML system that identifies service aspects mentioned in telecom customer feedback and predicts sentiment (Positive / Negative / Neutral) for each aspect.

## Overview

Telecom companies receive feedback across channels — app reviews, support tickets, surveys, social media. Traditional sentiment analysis gives only an overall score. This system provides **fine-grained, aspect-level** analysis:

**Input:** "The internet speed is good, but customer support is very poor."

**Output:**

| Aspect | Sentiment |
|--------|-----------|
| Internet Speed | Positive |
| Customer Support | Negative |

---

## Features

- **22 telecom aspect categories** — Network Coverage, Internet Speed, Call Quality, Billing, Roaming, 5G Experience, etc.
- **Multi-aspect detection** — handles feedback mentioning multiple services in one sentence
- **DeBERTa-based aspect detection** — fine-tuned transformer that understands context (replaces keyword-based approach)
- **Intelligent out-of-domain rejection** — correctly ignores non-telecom text
- **Interactive Streamlit UI** — single feedback analysis with visualizations
- **Batch CSV processing** — upload a CSV and get bulk predictions with dashboards
- **Confidence scoring** — probability estimates for each prediction
- **Domain-aware corrections** — telecom-specific lexicon fixes where generic models fail

---

## Model Architecture

### Aspect Detection: Fine-tuned DeBERTa-v3-base

```
Raw Feedback
    │
    ├── DeBERTa Tokenizer (max_length=256)
    │
    ├── DeBERTa-v3-base Encoder
    │
    ├── 22 Sigmoid Output Heads (multi-label classification)
    │
    ├── Threshold filtering (default: 0.5)
    │
    └── Detected Aspects: [aspect1, aspect2, ...]
```

**Aspect Classifier Metrics:**
- Micro F1: **97%+**
- Handles indirect language (buffering, streaming, lag → Internet Speed)
- Multi-label: detects multiple aspects per feedback

### Sentiment Prediction: Two Pipeline Options

The system supports two sentiment inference pipelines selectable from the app sidebar:

#### Pipeline 1: TF-IDF + Logistic Regression (Default)

```
Detected Aspects
    │
    ├── Clause Splitting ──── Contrastive conjunctions (but, however, though...)
    │
    ├── Sentiment Prediction
    │   ├── TF-IDF + Tuned Logistic Regression
    │   ├── VADER correction (layer 1)
    │   └── Domain lexicon (layers 2-4)
    │
    └── Results: [{aspect, sentiment, confidence}, ...]
```

**Sentiment Model:** Tuned Logistic Regression (C=0.5, lbfgs solver, L2 penalty)
- Test F1: **84.9%** (weighted)
- Trained on 13,100 aspect-level samples across 22 categories

#### Pipeline 2: Fine-tuned DistilBERT

```
Detected Aspects
    │
    ├── For each aspect:
    │   ├── Input: "[Aspect Name] relevant_text"
    │   ├── DistilBERT tokenizer (max_length=128)
    │   ├── DistilBertForSequenceClassification
    │   └── Softmax → sentiment + confidence
    │
    └── Results: [{aspect, sentiment, confidence}, ...]
```

**DistilBERT Model:** Fine-tuned `distilbert-base-uncased` for 3-class sentiment
- Test F1: **95.6%** (weighted)
- Processes raw text directly (no TF-IDF preprocessing needed)
- No VADER/domain corrections required — the transformer handles negation and implicit sentiment natively

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip
- Git LFS (required for transformer models)

### Installation

```bash
# Install Git LFS (if not already installed)
# Ubuntu/Debian:
sudo apt install git-lfs

# Initialize Git LFS
git lfs install

# Clone the repository
git clone https://github.com/AdvaithR12/ABSA-project.git
cd ABSA-project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (auto-downloads on first run, but can be done manually)
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('vader_lexicon')"
```

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
project/
│
├── app.py                      # Streamlit application (main entry point)
├── model_utils.py              # Core inference pipeline (aspect detection + sentiment)
├── evaluate_model.py           # Model evaluation against test datasets
├── config.yaml                 # Centralized configuration (thresholds, paths, params)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── project_report.md           # Detailed project report
│
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── config.py               # Configuration loader
│   ├── logger.py               # Centralized logging setup
│   ├── preprocessing.py        # Text preprocessing pipeline
│   ├── aspect_detection.py     # Aspect identification (DeBERTa + TF-IDF fallback)
│   ├── spell_correction.py     # Spelling correction with SymSpell
│   └── sentiment.py            # Sentiment prediction with corrections
│
├── scripts/                    # Utility scripts
│   ├── clean_aspect_labels.py  # Clean mislabeled training data
│   ├── train_aspect_classifier.py  # Train TF-IDF aspect classifier
│   └── add_edge_case_training_data.py  # Add edge cases for DeBERTa training
│
├── data/
│   ├── raw/                    # Raw datasets
│   │   ├── absa_dataset.csv    # Original dataset
│   │   └── raw_feedback.csv
│   ├── processed/              # Preprocessed & feature-engineered data
│   │   ├── absa_dataset_cleaned.csv  # Cleaned dataset (mislabels removed + edge cases)
│   │   ├── absa_processed.csv
│   │   └── absa_feature_engineered.csv
│   └── dataGenerator/          # Data generation scripts & prompts
│       ├── batchRunner.py      # Batch data generation via Groq API
│       ├── dataMerge.py        # Merge & validate generated batches
│       └── raw_batches/        # 136 generated JSON batch files
│
├── notebooks/
│   ├── EDA_RAW.ipynb           # Exploratory Data Analysis
│   ├── preprocess.ipynb        # Preprocessing pipeline
│   ├── feature_eng.ipynb       # Feature engineering (TF-IDF, SMOTE, encoding)
│   ├── models.ipynb            # Sentiment model training & comparison
│   └── train_deberta_aspect_classifier.ipynb  # DeBERTa aspect classifier training
│
├── models/                     # Trained model artifacts
│   ├── deberta_aspect/         # Fine-tuned DeBERTa aspect classifier
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── label_mapping.json
│   │   ├── metadata.json
│   │   └── training_args.bin
│   ├── distilbert_sentiment/   # Fine-tuned DistilBERT sentiment classifier
│   ├── best_sentiment_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── sgd_svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── sentiment_label_encoder.pkl
│   ├── aspect_classifier.pkl   # TF-IDF aspect classifier (fallback)
│   ├── aspect_mlb.pkl
│   ├── aspect_tfidf.pkl
│   └── versions/               # Versioned model snapshots
│
├── experiments/
│   └── experiment_log.json     # Experiment tracking (all training runs)
│
├── logs/                       # Runtime logs
│   ├── absa_inference.log
│   ├── aspect_distilbert_training.log
│   ├── batchRunner.log
│   └── merge.log
│
└── outputs/
    ├── eda/                    # EDA visualization outputs
    └── feature_engineering_comparison.png
```

---

## Dataset Creation

The dataset was synthetically generated using carefully designed prompts.

- **Total batches:** 136 (100 initial + 36 Neutral-focused augmentation)
- **Final dataset:** 6,750 unique feedback entries → 13,100 aspect-level rows
- **Cleaned dataset:** 11,734 rows (after removing mislabels + adding 368 edge cases)
- **Aspects:** 22 telecom categories
- **Sentiment distribution:** Positive 35.7% | Neutral 33.1% | Negative 31.2%
- **Characteristics:** Multi-aspect feedback, realistic slang, spelling errors, varied lengths

### Data Generation Prompt

The following prompt was used with the Groq API to generate each batch of 50 feedback entries:

```
You are a data generation assistant. Your job is to generate realistic synthetic telecom 
customer feedback data for training an Aspect-Based Sentiment Analysis (ABSA) model. 
Generate feedback that sounds like real customers writing about their telecom experience. 
Follow all instructions exactly and return only valid JSON.

Requirement: Generate 50 telecom customer feedback entries and return them as a JSON object 
containing two datasets ready for saving as CSV files.

ASPECT LIST (22 aspects — use EXACTLY these names):
Network Coverage, Internet Speed, Call Quality, Customer Support, Billing, Recharge Plans, 
Data Balance, Roaming, SIM Activation, Mobile App Experience, OTT Bundle Services, Pricing, 
Value for Money, Data Validity, 5G Experience, Network Outage, Number Portability, SMS Services, 
Postpaid Plans, Network Congestion, International Calling, Device Compatibility

SENTIMENT VALUES (use EXACTLY these):
Positive, Negative, Neutral

ASPECT FOCUS FOR THIS BATCH: [ASPECT_FOCUS]
SENTIMENT DISTRIBUTION FOR THIS BATCH: [SENTIMENT_MIX]
FEEDBACK IDs START FROM: fb_[START_ID]

FEEDBACK GENERATION RULES:
- 30% SHORT feedbacks (1 sentence, under 20 words)
- 40% MEDIUM feedbacks (2-3 sentences)  
- 30% LONG feedbacks (4-6 sentences with details)
- 15% include minor spelling mistakes or SMS-style language
- Include casual/angry/informal language naturally where appropriate
- 60% must mention MORE THAN ONE aspect
- Each aspect mentioned must have exactly one sentiment: Positive, Negative, or Neutral

OUTPUT FORMAT:
Return ONLY a valid JSON object with exactly this structure:

{
  "raw_feedback": [
    {
      "feedback_id": "fb_XXXXX",
      "feedback_text": "the customer feedback text here",
      "aspects": "Aspect1|Aspect2",
      "sentiments": "Positive|Negative",
      "num_aspects": 2
    }
  ],
  "absa_dataset": [
    {
      "feedback_id": "fb_XXXXX",
      "feedback_text": "the customer feedback text here",
      "aspect": "Aspect1",
      "sentiment": "Positive"
    }
  ]
}

RULES:
- raw_feedback must have exactly 50 entries (one per feedback)
- absa_dataset must have one entry per aspect per feedback
- aspects and sentiments in raw_feedback are pipe-separated (|) in the same order
- aspect names must match EXACTLY from the aspect list above
- sentiment must be EXACTLY one of: Positive, Negative, Neutral
```

The `[ASPECT_FOCUS]` and `[SENTIMENT_MIX]` placeholders were varied per batch to ensure coverage across all 22 aspects and balanced sentiment distribution. A separate neutral-focused batch runner was used for the final 35 batches to address initial Neutral-class underrepresentation.

### Generation Pipeline

1. `data/dataGenerator/batchRunner.py` — generates 50 entries per batch (not in repo - run separately)
2. `data/dataGenerator/raw_batches/` — Stores 136 raw JSON batch files
3. `data/dataGenerator/dataMerge.py` — Validates and merges all batches into final CSVs (`data/raw/absa_dataset.csv`, `data/raw/raw_feedback.csv`)

---

## Model Training

This section explains how to train all models from scratch.

### Prerequisites

- Python 3.10+ with dependencies installed (`pip install -r requirements.txt`)
- GPU recommended for transformer models (DistilBERT, DeBERTa)
- ~2GB disk space for model artifacts

### Training Pipeline Overview

```
data/raw/absa_dataset.csv
        │
        └──► notebooks/preprocess.ipynb
                    │
                    └──► data/processed/absa_processed.csv
                                │
                                └──► notebooks/feature_eng.ipynb
                                          │
                                          └──► data/processed/absa_feature_engineered.csv
                                                      │
                                                      └──► notebooks/models.ipynb
                                                                │
                                                                ├──► Logistic Regression
                                                                ├──► Naive Bayes
                                                                ├──► SGD-SVM
                                                                └──► DistilBERT (sentiment)

data/processed/absa_dataset_cleaned.csv
        │
        └──► notebooks/train_deberta_aspect_classifier.ipynb
                    │
                    └──► DeBERTa (aspect detection)
```

### Step 1: Preprocessing

Converts raw text to cleaned, tokenized, lemmatized format.

```bash
jupyter notebook notebooks/preprocess.ipynb
# Run all cells
# Output: data/processed/absa_processed.csv
```

**What it does:**
- Lowercase text
- Remove URLs, emails, special characters
- Tokenize with NLTK
- Remove stopwords (keeps negations: not, no, never)
- Lemmatize words
- Filter short tokens (≤2 chars)

### Step 2: Feature Engineering

Creates TF-IDF features and applies SMOTE for class balancing.

```bash
jupyter notebook notebooks/feature_eng.ipynb
# Run all cells
# Output: data/processed/absa_feature_engineered.csv
```

**What it does:**
- TF-IDF vectorization (unigrams + bigrams)
- Label encoding for sentiment classes
- Train/validation/test split (64/16/20) by feedback_id
- SMOTE oversampling for class balance

### Step 3: Train Sentiment Models

Trains and compares 4 sentiment classifiers.

```bash
jupyter notebook notebooks/models.ipynb
# Run all cells
# Output: models/*.pkl, models/distilbert_sentiment/
```

**Models trained:**

| Model | Output File | Training Time |
|-------|-------------|---------------|
| Logistic Regression | `best_sentiment_model.pkl` | ~3 sec |
| Naive Bayes | `naive_bayes_model.pkl` | ~0.01 sec |
| SGD-SVM | `sgd_svm_model.pkl` | ~0.6 sec |
| DistilBERT | `distilbert_sentiment/` | ~25 min |

**Also saves:**
- `tfidf_vectorizer.pkl` — TF-IDF vectorizer
- `sentiment_label_encoder.pkl` — Label encoder
- `models/versions/{timestamp}/` — Versioned snapshot

### Step 4: Train DeBERTa Aspect Classifier

Fine-tunes DeBERTa-v3-base for multi-label aspect detection.

#### Option A: Local Training (GPU required)

```bash
jupyter notebook notebooks/train_deberta_aspect_classifier.ipynb
# Run all cells
# Output: models/deberta_aspect/
```

#### Option B: Google Colab Training

1. Upload `notebooks/train_deberta_aspect_classifier.ipynb` to [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Upload `data/processed/absa_dataset_cleaned.csv` when prompted
4. Run all cells
5. Download the model folder and extract to `models/deberta_aspect/`

**Training configuration:**
- Epochs: 5
- Batch size: 16
- Learning rate: 2e-5
- Max sequence length: 256
- Training time: ~30-45 min on T4 GPU

**Output files:**
```
models/deberta_aspect/
├── model.safetensors      # Model weights
├── config.json            # Architecture config
├── tokenizer.json         # Tokenizer vocabulary
├── tokenizer_config.json  # Tokenizer settings
├── label_mapping.json     # Aspect index → name mapping
├── metadata.json          # Training metadata
└── training_args.bin      # Hyperparameters
```

### Quick Retrain Commands

**Retrain all sentiment models:**
```bash
cd notebooks
jupyter nbconvert --to notebook --execute preprocess.ipynb
jupyter nbconvert --to notebook --execute feature_eng.ipynb
jupyter nbconvert --to notebook --execute models.ipynb
```

**Retrain DeBERTa aspect classifier only:**
```bash
jupyter nbconvert --to notebook --execute notebooks/train_deberta_aspect_classifier.ipynb
```

### Data Sources

| Model | Training Data | Rows |
|-------|---------------|------|
| Sentiment models (LR, NB, SVM, DistilBERT) | `absa_processed.csv` | 13,326 |
| DeBERTa aspect classifier | `absa_dataset_cleaned.csv` | 11,734 |

The cleaned dataset has mislabeled rows removed and 368 edge cases added for better indirect language handling.

---

## Model Comparison

### Aspect Detection

| Model | Micro F1 | Handles Indirect Language | Out-of-Domain Rejection |
|-------|----------|---------------------------|------------------------|
| **DeBERTa-v3-base** | **97%+** | Yes | Yes |
| TF-IDF + Keywords | 93% | Limited | No |

### Sentiment Prediction

| Model | Test F1 | Train Time | Inference Time | Memory |
|-------|---------|-----------|---------------|--------|
| **DistilBERT (fine-tuned)** | **0.956** | ~25min (GPU) | ~50ms | 257 MB |
| SGD-SVM | 0.853 | 0.62s | 0.001s | 0.5 MB |
| Tuned Logistic Regression | 0.849 | 3.0s | 0.002s | 11.5 MB |
| Naive Bayes | 0.846 | 0.01s | 0.002s | 1.4 MB |

---

## Assumptions & Tradeoffs

1. **Synthetic data** — Dataset is LLM-generated, not from real telecom customers. May not capture all real-world language patterns.
2. **DeBERTa for aspects, TF-IDF option for sentiment** — DeBERTa provides superior aspect detection; TF-IDF sentiment is faster (~2ms) for CPU-only deployments.
3. **Threshold tuning** — Default 0.5 confidence threshold balances precision/recall. Can be adjusted per use case.
4. **Single sentiment per aspect** — The system assigns one sentiment per detected aspect. Truly "Mixed" feedback on a single aspect is not natively supported.
5. **Fallback mechanism** — If DeBERTa is unavailable (missing files, no torch), system falls back to TF-IDF + keyword validation.

---

## Future Improvements

- **Emoji sentiment support** — Detect and interpret emojis (👍, 😡, 🔥, etc.) as sentiment signals, which are common in real customer feedback
- **Multilingual aspect identification** — Support feedback in multiple languages (Hindi, Hinglish, regional languages) for broader Indian telecom coverage
- **Learned sentiment lexicon** — Auto-extract domain polarity scores from training data
- **Multi-task model** — Joint aspect detection + sentiment in a single architecture
- **Real customer data** — Replace synthetic data with production feedback for better generalization
- **A/B testing framework** — Compare model versions on live traffic


---

## Experiment Tracking

All experiments are logged in `experiments/experiment_log.json` including:
- Hyperparameters tested
- Dataset versions used
- Train/val/test metrics
- Notes on what worked and what didn't

---

## License

This project was created as a Data Science capstone submission.
