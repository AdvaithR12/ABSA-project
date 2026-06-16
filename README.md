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
- **ML + keyword hybrid aspect detection** — learned classifier catches implicit mentions that keywords miss
- **Interactive Streamlit UI** — single feedback analysis with visualizations
- **Batch CSV processing** — upload a CSV and get bulk predictions with dashboards
- **Confidence scoring** — probability estimates for each prediction
- **Domain-aware corrections** — telecom-specific lexicon fixes where generic models fail

---

## Model Architecture

The system supports two inference pipelines selectable from the app sidebar:

### Pipeline 1: TF-IDF + Logistic Regression (Default)

```
Raw Feedback
    │
    ├── Aspect Detection ──────────────────┐
    │   ├── Keyword regex matching         │
    │   └── ML classifier (OneVsRest LR)   │
    │                                      │
    ├── Clause Splitting ─────────────────────── Contrastive conjunctions (but, however, though...)
    │
    ├── Sentiment Prediction ──────────────┐
    │   ├── TF-IDF + Tuned Logistic Reg.   │
    │   ├── VADER correction (layer 1)     │
    │   └── Domain lexicon (layers 2-4)    │
    │                                      │
    └── Results: [{aspect, sentiment, confidence}, ...]
```

**Sentiment Model:** Tuned Logistic Regression (C=0.5, saga solver, L2 penalty)
- Test F1: **84.4%** (weighted)
- Trained on 13,100 aspect-level samples across 22 categories

### Pipeline 2: Fine-tuned DistilBERT

```
Raw Feedback
    │
    ├── Aspect Detection (same as above)
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
- Test F1: **95.7%** (weighted)
- Processes raw text directly (no TF-IDF preprocessing needed)
- No VADER/domain corrections required — the transformer handles negation and implicit sentiment natively
- Requires `torch` and `transformers` packages

### Aspect Classifier (shared by both pipelines)

**OneVsRest Logistic Regression** with trigram TF-IDF
- Micro F1: **93.0%**
- Catches implicit aspect mentions ("mobile application" → Mobile App Experience)

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip
- Git LFS (required for the DistilBERT model)

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

### Running the Evaluation Script

```bash
python3 evaluate_model.py
```

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
├── model_analysis.md           # Model analysis & error report
│
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── config.py               # Configuration loader
│   ├── logger.py               # Centralized logging setup
│   ├── preprocessing.py        # Text preprocessing pipeline
│   ├── aspect_detection.py     # Aspect identification (keywords + ML)
│   └── sentiment.py            # Sentiment prediction with corrections
│
├── data/
│   ├── raw/                    # Raw datasets (absa_dataset.csv, raw_feedback.csv)
│   ├── processed/              # Preprocessed & feature-engineered data
│   └── dataGenerator/          # Data generation scripts & prompts
│       ├── dataPrompt.md       # LLM prompt used for dataset generation
│       ├── batchRunner.py      # Batch data generation via Groq API
│       ├── dataMerge.py        # Merge & validate generated batches
│       └── raw_batches/        # 135 generated JSON batch files
│
├── notebooks/
│   ├── EDA_RAW.ipynb           # Exploratory Data Analysis
│   ├── preprocess.ipynb        # Preprocessing pipeline
│   ├── feature_eng.ipynb       # Feature engineering (TF-IDF, SMOTE, encoding)
│   └── models.ipynb            # Model training, comparison & aspect classifier
│
├── models/                     # Trained model artifacts
│   ├── best_sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── sentiment_label_encoder.pkl
│   ├── naive_bayes_model.pkl
│   ├── sgd_svm_model.pkl
│   ├── calibrated_sentiment_model.pkl
│   ├── aspect_classifier.pkl
│   ├── aspect_mlb.pkl
│   ├── aspect_tfidf.pkl
│   └── distilbert_sentiment/  # Fine-tuned DistilBERT (config, weights, tokenizer)
│
├── experiments/
│   └── experiment_log.json     # Experiment tracking (all training runs)
│
├── logs/                       # Runtime logs
│   └── absa_inference.log
│
└── outputs/
    └── eda/                    # EDA visualization outputs
```

---

## Dataset Creation

The dataset was synthetically generated using carefully designed prompts.

- **Total batches:** 135 (100 initial + 35 Neutral-focused augmentation)
- **Final dataset:** 6,750 unique feedback entries → 13,100 aspect-level rows
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

1. `data/dataGenerator/batchRunner.py` — generates 50 entries per batch
2. `data/dataGenerator/raw_batches/` — Stores 135 raw JSON batch files
3. `data/dataGenerator/dataMerge.py` — Validates and merges all batches into final CSVs (`data/raw/absa_dataset.csv`, `data/raw/raw_feedback.csv`)

---

## Training Steps

1. **EDA** (`notebooks/EDA_RAW.ipynb`) — dataset statistics, distributions, imbalance analysis
2. **Preprocessing** (`notebooks/preprocess.ipynb`) — lowercasing, tokenization, lemmatization, stopword removal
3. **Feature Engineering** (`notebooks/feature_eng.ipynb`) — TF-IDF vectorization, label encoding, SMOTE, train/test split
4. **Model Training** (`notebooks/models.ipynb`) — trains 4 models (LR, NB, SGD-SVM, DistilBERT), hyperparameter tuning, comparison, aspect classifier

To retrain from scratch:
```bash
# Run notebooks in order (or use jupyter)
jupyter notebook notebooks/preprocess.ipynb
jupyter notebook notebooks/feature_eng.ipynb
jupyter notebook notebooks/models.ipynb
```

---

## Model Comparison

| Model | Test F1 | Train Time | Inference Time | Memory |
|-------|---------|-----------|---------------|--------|
| **DistilBERT (fine-tuned)** | **0.957** | ~25min (GPU) | ~50ms | 257 MB |
| Tuned Logistic Regression | 0.844 | 6.75s | 0.002s | 11.6 MB |
| Naive Bayes | 0.840 | 0.01s | 0.002s | 1.4 MB |
| SGD-SVM | 0.826 | 0.55s | 0.001s | 0.5 MB |

DistilBERT achieves the highest accuracy but requires GPU for training and has slower inference. Tuned LR is the default model — best balance of accuracy, speed, and no GPU requirement.

---

## Assumptions & Tradeoffs

1. **Synthetic data** — Dataset is LLM-generated, not from real telecom customers. May not capture all real-world language patterns.
2. **TF-IDF over transformers** — Chosen for fast inference (~2ms) and no GPU requirement. Tradeoff: weaker on implicit/indirect sentiment.
3. **Keyword + ML hybrid** — Keywords provide explainability and speed; ML catches implicit mentions. Tradeoff: keyword dict needs manual maintenance.
4. **Domain lexicon as stopgap** — Regex-based phrase matching for domain-specific sentiment. Would be replaced by fine-tuned transformer in a GPU-equipped environment.
5. **Single sentiment per aspect** — The system assigns one sentiment per detected aspect. Truly "Mixed" feedback on a single aspect is not natively supported.

---

## Future Improvements

- **Emoji sentiment support** — Detect and interpret emojis (👍, 😡, 🔥, etc.) as sentiment signals, which are common in real customer feedback
- **Multilingual aspect identification** — Support feedback in multiple languages (Hindi, Hinglish, regional languages) for broader Indian telecom coverage
- **Learned sentiment lexicon** — Auto-extract domain polarity scores from training data
- **Multi-task model** — Joint aspect detection + sentiment in a single architecture
- **Real customer data** — Replace synthetic data with production feedback for better generalization
- **A/B testing framework** — Compare model versions on live traffic
- **DistilBERT for aspect detection** — Replace the keyword + OneVsRest ML hybrid with a transformer-based aspect extractor

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
