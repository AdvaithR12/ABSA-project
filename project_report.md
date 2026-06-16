# Aspect-Based Sentiment Analysis (ABSA) — Project Report

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Pipeline](#data-pipeline)
3. [EDA Observations](#eda-observations)
4. [Preprocessing Decisions](#preprocessing-decisions)
5. [Feature Engineering](#feature-engineering)
6. [Model Development](#model-development)
   - 6.1 [Models Selected](#61-models-selected)
   - 6.2 [Comparison Results](#62-comparison-results)
   - 6.3 [Best Model: DistilBERT](#63-best-model-distilbert)
   - 6.4 [Default Model: Tuned Logistic Regression](#64-default-model-tuned-logistic-regression)
   - 6.5 [Why Each Model Was Selected](#65-why-each-model-was-selected)
   - 6.6 [DistilBERT Fine-tuning Details](#66-distilbert-fine-tuning-details)
   - 6.7 [ML-Based Aspect Classifier](#67-ml-based-aspect-classifier)
   - 6.8 [Runtime Inference Enhancements](#68-runtime-inference-enhancements)
   - 6.9 [Model Limitations](#69-model-limitations)
   - 6.10 [Model Files Summary](#610-model-files-summary)
7. [Issues Identified & Fixes Applied](#issues-identified--fixes-applied)
8. [Error Analysis](#error-analysis)
9. [Final Results](#final-results)
10. [Recommendations](#recommendations)

---

## 1. Project Overview

**Objective:** Build an Aspect-Based Sentiment Analysis system for telecom customer feedback that can:
1. Identify relevant aspects (topics) from raw customer feedback
2. Predict sentiment (Positive / Negative / Neutral) for each identified aspect

**Dataset:** Synthetically generated telecom customer feedback covering 22 aspect categories across 6,750 unique feedback entries (13,100 aspect-level rows).

**Best Model:** Fine-tuned DistilBERT achieving **95.7% weighted F1** on the test set.
**Default Model:** Tuned Logistic Regression achieving **84.4% weighted F1** (fast inference, no GPU dependency).

---

## 2. Data Pipeline

### Data Generation
- 135 total batches generated via Groq API (100 initial + 35 Neutral-focused augmentation)
- Each batch: 50 feedback entries with multiple aspects per feedback
- Sentiment mix varied across batches (balanced, mostly negative, mostly positive)

### Data Merge
- `dataMerge.py` reads all `batch_*.json` files from `raw_batches/`
- Validates every entry (aspect names, sentiment values, required fields)
- Outputs two CSVs: `raw_feedback.csv` (one row per feedback) and `absa_dataset.csv` (one row per aspect)
- Deduplicates and logs invalid entries

### Pipeline Order
```
Data Generation → Merge → EDA → Preprocessing → Feature Engineering → Modeling
```

---

## 3. EDA Observations

### Dataset Statistics
- Total feedback entries: 6,750
- Total aspect-level rows: 13,100
- Unique aspects: 22
- Average aspects per feedback: 1.94
- Multi-aspect feedbacks: ~55%

### Sentiment Distribution (after neutral data addition)
| Sentiment | Count | Percentage |
|-----------|-------|-----------|
| Positive | 4,682 | 35.7% |
| Neutral | 4,332 | 33.1% |
| Negative | 4,086 | 31.2% |

### Aspect Distribution
- Most frequent: Call Quality (662)
- Least frequent: Number Portability (502)
- Aspect imbalance ratio: 1.32x (healthy — no aspect is severely underrepresented)

### Text Length Statistics
- Mean word count: ~25 words per feedback
- Range: 5-80+ words
- 30% short (1 sentence), 40% medium (2-3 sentences), 30% long (4-6 sentences)

### Key EDA Finding
- Original sentiment distribution was **severely imbalanced**: Neutral at 13.1% vs Positive 46.4% and Negative 40.5%
- This directly caused model failure on the Neutral class

---

## 4. Preprocessing Decisions

### Steps Applied
1. **Lowercase conversion** — normalize case
2. **URL/email removal** — irrelevant noise
3. **Special character removal** — keep only alphanumeric + spaces
4. **Whitespace normalization** — collapse multiple spaces
5. **Tokenization** — NLTK word_tokenize
6. **Stopword removal** — standard English stopwords MINUS negation words (not, no, nor, never)
7. **Lemmatization** — WordNet lemmatizer to reduce morphological variants
8. **Noise word removal** — domain-specific slang (lol, bruh, fr, u, ur, pls, plz)
9. **Empty row filtering** — drop rows where processed_text is empty

### Rationale for Keeping Negation Words
Negation words are critical for sentiment: "not good" vs "good" have opposite meanings. Standard stopword lists include these, so we explicitly retain them.

### Rationale for Removing Slang
Words like "lol", "bruh", "fr" appear inconsistently and don't carry reliable sentiment signal. They add noise without predictive value.

---

## 5. Feature Engineering

### Label Encoding
- Sentiment → integer: Negative=0, Neutral=1, Positive=2
- LabelEncoder saved to `models/sentiment_label_encoder.pkl`

### TF-IDF Vectorization
- Unigrams + bigrams: `ngram_range=(1, 2)`
- Minimum document frequency: `min_df=2` (terms must appear in at least 2 documents)
- Maximum document frequency: `max_df=0.95` (terms in >95% of docs are too common)
- Sublinear TF: `sublinear_tf=True` (log-scaled term frequency to reduce dominance of very frequent terms)
- Vocabulary size: ~12,000 features

### Aspect Feature (added during modeling)
- Combined input: `aspect_name + processed_text` (e.g., "network_coverage network terrible area signal drop")
- This tells the model WHICH aspect the sentiment applies to — making it true ABSA

### SMOTE Oversampling
- Applied **only to Naive Bayes** training to equalize the minority class (all classes → 2,997 samples each)
- Logistic Regression and SGD-SVM use `class_weight='balanced'` instead (adjusts loss function weights inversely proportional to class frequency)
- These are alternative strategies for handling class imbalance — they are not combined on the same model

| Model | Balancing Strategy |
|-------|-------------------|
| Naive Bayes | SMOTE (synthetic oversampling on training data) |
| Logistic Regression | `class_weight='balanced'` (cost-sensitive learning) |
| SGD-SVM | `class_weight='balanced'` (cost-sensitive learning) |
| DistilBERT | None (near-balanced distribution; cross-entropy loss) |


### Train/Test Split
- 80% train / 20% test (stratified)
- Further 80/20 split on training: 8,384 train / 2,096 validation / 2,620 test

---

## 6. Model Development

### 6.1. Models Selected

| Model | Type | Why Selected |
|-------|------|-------------|
| Logistic Regression | Linear classifier | Strong baseline for sparse text features, interpretable, supports class_weight |
| Multinomial Naive Bayes | Probabilistic | Designed for count/frequency features, extremely fast training |
| SGD-SVM (modified_huber) | Linear SVM via SGD | Scales well, no convergence issues, probability-capable |
| DistilBERT (fine-tuned) | Transformer | Contextual embeddings handle negation, word order, and implicit sentiment natively |

### 6.2. Comparison Results (Test Set)

+--------------+----------+--------+---------+---------------+----------------+--------+
| Model        | Train F1 | Val F1 | Test F1 | Train Time    | Inference Time | Memory |
+--------------+----------+--------+---------+---------------+----------------+--------+
| DistilBERT   | 0.970    | 0.960  | 0.957   | ~25 min (GPU) | ~50 ms         | 257 MB |
| Tuned LR     | 0.880    | 0.838  | 0.844   | 7.7 s         | 0.002 s        | 11.6 MB|
| Naive Bayes  | 0.864    | 0.841  | 0.840   | 0.01 s        | 0.002 s        | 1.4 MB |
| SGD-SVM      | 0.909    | 0.812  | 0.826   | 0.57 s        | 0.001 s        | 0.5 MB |
+--------------+----------+--------+---------+---------------+----------------+--------+

### 6.3. Best Model: DistilBERT (F1: 95.7%)
- Fine-tuned `distilbert-base-uncased` with 3-class sentiment head
- Input format: `[Aspect Name] raw feedback text` (max 128 tokens)
- Handles negation, implicit sentiment, and sarcasm natively
- Requires `torch` and `transformers` packages; ~50ms inference per aspect

### 6.4. Default Model: Tuned Logistic Regression (F1: 84.4%)
- Parameters: C=0.5, solver='lbfgs', penalty='l2', max_iter=3000, class_weight='balanced'
- Best CV F1: 0.841
- Test F1: 0.844 (slight negative gap = no overfitting)
- Used as default due to fast inference (~2ms) and no GPU/torch dependency

### 6.5. Why Each Model Was Selected

**Logistic Regression:** Strong baseline for text classification. Handles high-dimensional sparse TF-IDF features efficiently. Supports `class_weight='balanced'` natively. Highly interpretable — feature coefficients reveal which words drive each sentiment.

**Naive Bayes (MultinomialNB):** Specifically designed for count/frequency-based features like TF-IDF. Extremely fast training (0.01s). Serves as a lightweight baseline.

**SGD-SVM (modified_huber loss):** Equivalent to a linear SVM but scales better via stochastic gradient descent. No convergence issues unlike standard `LinearSVC`. Provides probability estimates while maintaining SVM-like decision boundaries.

**DistilBERT:** Contextual embeddings capture word order, negation, and implicit sentiment — all weaknesses of TF-IDF models. Despite the smaller dataset (13,100 samples), the transformer significantly outperformed linear models (95.7% vs 84.4% F1).

### 6.6. DistilBERT Fine-tuning Details

- **Base model:** `distilbert-base-uncased` (66M parameters, 6 transformer layers)
- **Task head:** Linear classifier for 3-class sentiment (Negative, Neutral, Positive)
- **Input format:** `[Aspect Name] raw feedback text` (max 128 tokens)
- **No preprocessing required** — handles punctuation, case, and word order internally
- **Training data:** Same 13,100 aspect-level samples with aspect prefix
- **Optimizer:** AdamW
- **Training time:** ~25 minutes on GPU
- **Model size:** 257 MB (stored via Git LFS)

**Integration:** Selectable from the app sidebar. When selected, `predict_absa()` routes to `predict_absa_bert()` automatically. No VADER/domain corrections applied — the transformer's predictions are used directly.

---

### 6.7. ML-Based Aspect Classifier

#### Problem
The keyword-based aspect detection (`ASPECT_KEYWORDS`) missed implicit mentions:
- "The mobile application takes too long to load" → General (should be Mobile App Experience)
- "Number transfer was smooth" → General (should be Number Portability)
- "Text messages arrive hours late" → General (should be SMS Services)
- "Data runs out too quickly" → General (should be Data Balance)

#### Solution
A multi-label OneVsRest Logistic Regression classifier trained on the 13,100-sample dataset:
- **Input:** TF-IDF vectors with trigrams (ngram_range 1-3, max_features=15,000)
- **Output:** Binary prediction for each of 22 aspects (multi-label)
- **Architecture:** OneVsRestClassifier wrapping LogisticRegression (C=2.0, balanced class weights)
- **Training data:** 6,750 unique feedback samples grouped by feedback_id with multi-label aspect lists

#### Performance
| Metric | Score |
|--------|-------|
| Micro F1 | 0.930 |
| Macro F1 | 0.930 |
| Samples F1 | 0.958 |


#### Edge Case Results
| Feedback | Keywords | ML Classifier |
|----------|----------|---------------|
| "The mobile application takes too long to load" | General | Mobile App Experience (0.89) |
| "Number transfer was smooth and hassle-free" | General | Number Portability (0.79) |
| "Text messages arrive hours late" | General | SMS Services (0.88) |
| "The porting process took forever" | General | Number Portability (0.95) |
| "Data runs out too quickly" | General | Data Balance (0.99) |

---

### 6.8. Runtime Inference Enhancements

Post-training enhancements in the prediction pipeline (`model_utils.py`) that address the TF-IDF model's weaknesses without modifying the trained sentiment model.

#### Clause-Level Splitting
Splits input on contrastive conjunctions (`but`, `however`, `though`, `although`, `yet`, `while`, `whereas`, `nevertheless`) so each aspect is matched only to the clause(s) that mention it. Adjacent clause inclusion handles cases where the contrastive clause implicitly refers to the same aspect.

#### VADER Sentiment Correction (Layer 1)
VADER (rule-based lexicon) corrects when model confidence is low (<60%):
- VADER ≤ -0.3 → override to Negative
- VADER ≥ +0.3 → override to Positive

#### Domain-Aware Sentiment Lexicon (Layers 2-4)
A telecom-specific phrase lexicon with regex patterns catches language VADER scores as neutral:
- **16 positive patterns:** "seamless", "hassle-free", "completed within [time]", "resolved quickly", etc.
- **21 negative patterns:** "too long/slow/expensive", "took forever", "crashes", "never resolved", "no signal", etc.

| Layer | Trigger | Action |
|-------|---------|--------|
| Layer 2 | Model=Neutral, confidence<0.65, domain has signal | Override to Positive/Negative |
| Layer 3 | Model=Neutral, confidence<0.75, domain signal strong | Override without VADER agreement |
| Layer 4 | Model=Positive, confidence<0.60, domain=negative | Override to Negative |

#### Evaluation Results (51 samples)

| Metric | Before Enhancements | After Enhancements |
|--------|--------------------|--------------------|
| Exact match accuracy | 66.7% (34/51) | **94.1% (48/51)** |
| Relaxed accuracy | 73.5% | **97.1%** |
| "General" bucket | Present | **Eliminated** |
| Negative recall | 68.8% | **100%** |
| Positive recall | 82.6% | **100%** |

#### Design Rationale
- **Non-destructive:** The trained model is never modified. Enhancements are purely at inference time.
- **Layered and conservative:** Each correction layer has an increasing confidence threshold (0.60 → 0.65 → 0.75).
- **Fallback-safe:** If ML models don't exist on disk, the system gracefully falls back to keyword-only aspect detection.
- **Not applied to DistilBERT:** The transformer handles all these patterns natively, so corrections are only used with sklearn models.

---

### 6.9. Model Limitations

#### TF-IDF Sentiment Model Limitations
1. **Single-sentence multi-aspect ambiguity** — When two aspects with opposing sentiments appear in the same sentence, the TF-IDF model cannot isolate which words apply to which aspect.
2. **Negation handling is limited** — TF-IDF doesn't understand word order. "Not good" and "good not" produce the same features.
3. **Sarcasm and irony** — "Great, another outage. Exactly what I needed today." reads as positive on surface features.
4. **Out-of-vocabulary (OOV) inputs** — New slang or terms not seen during training get zero TF-IDF weight.
5. **Domain-specific only** — Won't generalize to other domains without retraining.
6. **Implicit sentiment** — Phrases like "took forever", "runs out quickly" lack explicit sentiment words.

#### Aspect Identification Limitations
1. **Keyword-dependent** — Regex patterns require manual updates for new terminology.
2. **Implicit aspect detection gaps** — ML classifier mitigates but doesn't fully solve.
3. **Over-detection on generic words** — "plan", "call", "speed" may trigger aspects in non-relevant contexts.

#### DistilBERT Limitations
1. **Inference speed** — ~50ms per aspect vs ~2ms for TF-IDF models.
2. **Model size** — 257 MB vs ~5 MB for all sklearn models combined.
3. **Dependency weight** — Requires `torch` (~2GB) and `transformers` packages.
4. **Still domain-specific** — Trained only on telecom data; won't generalize without retraining.

---

### 6.10. Model Files Summary

| File | Purpose | Size |
|------|---------|------|
| `models/best_sentiment_model.pkl` | Tuned LR sentiment classifier (C=0.5, lbfgs, l2) | 284 KB |
| `models/tfidf_vectorizer.pkl` | TF-IDF vectorizer (bigrams, ~12k vocab) | 336 KB |
| `models/sentiment_label_encoder.pkl` | LabelEncoder (Negative=0, Neutral=1, Positive=2) | <1 KB |
| `models/naive_bayes_model.pkl` | Alternative NB model | 568 KB |
| `models/sgd_svm_model.pkl` | Alternative SGD-SVM model | 284 KB |
| `models/calibrated_sentiment_model.pkl` | Calibrated model (experimental, reverted) | 288 KB |
| `models/aspect_classifier.pkl` | Multi-label aspect classifier (OneVsRest LR) | 2.1 MB |
| `models/aspect_mlb.pkl` | MultiLabelBinarizer (22 classes) | 4 KB |
| `models/aspect_tfidf.pkl` | Aspect TF-IDF (trigrams, 15k vocab) | 556 KB |
| `models/distilbert_sentiment/` | Fine-tuned DistilBERT (config, weights, tokenizer) | 257 MB |

---

## 7. Issues Identified & Fixes Applied

### Issue 1: Severe Class Imbalance (Neutral at 13%)

**Observation:** Original dataset had Neutral class at only 13.1%. Model achieved 23% recall on Neutral — effectively ignoring it.

**Root Cause:** Training data distribution. With Positive at 46% and Negative at 40%, the model learned to predict only those two classes confidently.

**Fix Applied:**
- Generated 35 additional batches (batch_101 to batch_135) focused entirely on Neutral sentiment
- Used programmatic generation with templates covering all 22 aspects
- Added ~3,015 new Neutral ABSA entries
- New distribution: Positive 35.7%, Neutral 33.1%, Negative 31.2%

**Result:** Neutral recall improved from 23% → **85.8%**

---

### Issue 2: Aspect Feature Missing

**Observation:** Original model only used `processed_text` as input — it was a general sentiment classifier, not ABSA.

**Root Cause:** The aspect column was available in the dataset but never used as a model feature.

**Fix Applied:**
- Created combined input: `aspect_name_lowercase + " " + processed_text`
- Example: "network_coverage network terrible area signal drop"
- The aspect token gives the model context about WHAT the sentiment applies to

**Result:** Model can now differentiate sentiment for different aspects in the same feedback text.

---

### Issue 3: Overfitting (Linear SVM)

**Observation:** Original LinearSVC had Train F1 of 87.4% vs Test F1 of 72.7% — a 15-point gap.

**Root Cause:** Default LinearSVC with C=1 on SMOTE data was too complex. Also had convergence warnings.

**Fix Applied:**
- Switched to SGDClassifier with `modified_huber` loss
- Increased `max_iter` to 5000
- Used `class_weight='balanced'` instead of SMOTE for SVM
- Broader hyperparameter search with RandomizedSearchCV

**Result:** SGD-SVM Train-Val gap of 9.7% (better than 15%), and Tuned LR has only 4.2% gap.

---

### Issue 4: No Per-Class Metrics Reported

**Observation:** Original model only reported weighted averages — hiding the collapsed Neutral class.

**Fix Applied:**
- Added `classification_report` with per-class precision/recall/F1
- Added confusion matrices for all splits
- Added dedicated "Neutral Class Performance" section

**Result:** Full visibility into each class's performance.

---

### Issue 5: Inconsistent SMOTE Application

**Observation:** Original Naive Bayes was trained WITHOUT SMOTE while LR and SVM used it — unfair comparison.

**Fix Applied:**
- All models now consistently use the same data (SMOTE or class_weight as appropriate)
- NB uses SMOTE data, LR uses class_weight='balanced', SGD uses class_weight='balanced'
- Comparison table shows fair apples-to-apples results

---

### Issue 6: Aspect Identifier False Positives

**Observation:** Keyword-based aspect detection used substring matching. "port" in "support" triggered Number Portability. "app" in "happens" triggered Mobile App.

**Root Cause:** Simple `if keyword in text` doesn't respect word boundaries.

**Fix Applied:**
- Replaced all keywords with word-boundary regex patterns: `r'\bport\b'` won't match "support"
- Removed ambiguous keywords ("port" from SIM Activation, "switch" from Number Portability)
- Made multi-word phrases more specific ("international call" → `r'\binternational call\b'`)

**Result:** Aspect precision improved from 54% → **72.5%**, F1 from 68.6% → **78.0%**

---

### Issue 7: Sentiment Cross-Contamination in Multi-Aspect Feedback

**Observation:** "Network coverage is pathetic but call quality is decent" → model saw full text for both aspects, causing incorrect predictions (positive words influenced negative aspect and vice versa).

**Root Cause:** Full feedback text was used for every aspect, regardless of which sentence mentions which aspect.

**Fix Applied:**
- Implemented `get_relevant_sentences()` — splits feedback into sentences using `sent_tokenize()`
- For each aspect, only the sentence(s) containing that aspect's keywords are used
- Falls back to full text only if no sentence matches (shouldn't happen)

**Result:** Multi-sentence feedback now correctly predicts different sentiments for different aspects (e.g., "5G blazing fast in city" → Positive, "Coverage drops in suburbs" → Negative).

---


### Issue 8: Narrow Hyperparameter Search

**Observation:** Original GridSearchCV only tested 4 values of C with a single solver. Best params were defaults.

**Fix Applied:**
- Switched to RandomizedSearchCV with 20 iterations
- Wider C range: [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
- Multiple solvers: lbfgs, saga
- Both balanced and unweighted class_weight tested
- 5-fold stratified cross-validation

**Result:** Found C=0.5 with lbfgs solver — marginal improvement but confirms the model is well-tuned.

---

## 8. Error Analysis

### Prediction Accuracy on Unseen Inputs: 80% (12/15 correct)

### Common Failure Patterns

| # | Pattern | Example | Expected | Predicted | Root Cause |
|---|---------|---------|----------|-----------|------------|
| 1 | Subtle negative phrasing | "Data balance runs out too fast" | Negative | Positive | "fast" has positive associations; "runs out" isn't strongly negative in training vocab |
| 2 | Mild positive as neutral | "Ported my number, smooth and quick" | Positive | Neutral | Model conservative — mild positives classified neutral due to overlapping training language |
| 3 | Intra-sentence mixed sentiment | "Pathetic coverage but decent call quality" | Coverage: Neg, Call: Neutral | Swapped | Single sentence — sentence splitter can't isolate clauses |
| 4 | Keyword gap | "International calling rates are high" | International Calling | Value for Money | Pattern `\binternational call\b` doesn't match "calling" |

### Confusing Category Pairs

1. **Neutral ↔ Positive** — "okay", "decent", "works fine", "no complaints" sit on the boundary. Model treats these inconsistently.
2. **Negative ↔ Neutral** — Mild complaints ("could be better", "nothing special") sometimes classified as Neutral.
3. **Pricing ↔ Value for Money** — Overlapping vocabulary. "Expensive but worth it" triggers both.
4. **Recharge Plans ↔ Postpaid Plans** — "plan" keyword appears in both; context needed to disambiguate.

### Sentiment Misclassification Patterns

1. **Negation blindness** (~5% of errors) — "Not bad" gets Negative weight from "bad". TF-IDF loses word order.
2. **Intensity insensitivity** — "okay" and "incredible" both map to Positive sometimes. No intensity gradation.
3. **Sarcasm/irony** — "Great, another outage. Exactly what I needed." reads as Positive on surface features.
4. **Implicit sentiment** — "Plan costs 500 for 1GB" is Negative by implication but has no sentiment words.
5. **Comparative confusion** — "Better than before but still not great" has mixed signals.

### Multi-Aspect Challenges

1. **Same-sentence mixed sentiment** — TF-IDF cannot attribute words to specific aspects within one sentence
2. **Aspect over-detection** — Generic words ("plan", "call", "speed") trigger aspects in non-relevant contexts
3. **Cascading errors** — Wrong aspect → wrong sentiment (aspect token is part of model input)

---

## 9. Final Results

### Per-Class Test Performance (Best Model: Tuned LR)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Negative | 0.789 | 0.842 | 0.815 |
| Neutral | 0.917 | 0.858 | 0.887 |
| Positive | 0.830 | 0.831 | 0.831 |

## 10. Recommendations

### Immediate Improvements
1. Expand International Calling keywords: add `r'\binternational calling\b'`
2. Add "runs out", "depletes", "drains" to negative training examples for Data Balance
3. Build a "commonly confused" validation set for continuous monitoring
4. **Emoji sentiment support** — interpret emojis as sentiment signals (👍 → Positive, 😡 → Negative)
5. **Multilingual aspect identification** — support Hindi, Hinglish, and regional language feedback

### Medium-Term
1. Replace TF-IDF with sentence-transformer embeddings for the linear models
2. Train aspect-specific sentiment classifiers (22 binary models) for higher per-aspect accuracy
3. Use DistilBERT for aspect detection as well (replace keyword + OneVsRest hybrid)

### Long-Term
1. Use LLM for zero-shot aspect extraction (eliminates keyword dictionary maintenance)
2. Multi-task architecture: joint aspect identification + sentiment prediction
3. Deploy with A/B testing against current model for incremental improvement validation
4. Replace synthetic data with real customer feedback for better generalization
