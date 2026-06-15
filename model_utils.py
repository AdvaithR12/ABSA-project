"""
ABSA Model Utilities
---
Contains all model-related logic: loading, preprocessing, aspect identification,
prediction pipeline, and VADER correction.

This module serves as the main interface used by the Streamlit app.
For modular source code, see src/ package.
"""

import re
import os
import logging
import numpy as np
import joblib
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# --- Logging Setup ---
logger = logging.getLogger('absa.model_utils')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    _console = logging.StreamHandler()
    _console.setFormatter(_formatter)
    logger.addHandler(_console)
    # File handler
    os.makedirs('logs', exist_ok=True)
    _fh = logging.FileHandler('logs/absa_inference.log', mode='a')
    _fh.setFormatter(_formatter)
    logger.addHandler(_fh)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)


# --- Constants ---
ASPECT_KEYWORDS = {
    'Network Coverage': [r'\bnetwork\b', r'\bcoverage\b', r'\bsignal\b', r'\bbars?\b', r'\bdead zone\b', r'\btower\b'],
    'Internet Speed': [r'\bspeed\b', r'\binternet\b', r'\bmbps\b', r'\bdownload\b', r'\bupload\b', r'\bbandwidth\b', r'\bbuffering\b'],
    'Call Quality': [r'\bcall quality\b', r'\bcall clarity\b', r'\bvoice\b', r'\bdropped call\b', r'\bcall drop\b', r'\becho\b', r'\bvolte\b', r'\bhd calling\b', r'\bcall\b'],
    'Customer Support': [r'\bcustomer support\b', r'\bcustomer care\b', r'\bcustomer service\b', r'\bhelpline\b', r'\bagent\b', r'\bcomplaint\b', r'\bticket\b', r'\bivr\b', r'\bsupport\b'],
    'Billing': [r'\bbilling\b', r'\bbill\b', r'\binvoice\b', r'\bcharge[sd]?\b', r'\bpayment\b', r'\bauto.?pay\b', r'\bdebit\b'],
    'Recharge Plans': [r'\brecharge\b', r'\bplan\b', r'\bpack\b', r'\bprepaid\b', r'\btopup\b'],
    'Data Balance': [r'\bdata balance\b', r'\bdata limit\b', r'\bdata usage\b', r'\bremaining data\b', r'\bdata cap\b', r'\b\d+\s*gb\b'],
    'Roaming': [r'\broaming\b', r'\btravel\b', r'\babroad\b'],
    'SIM Activation': [r'\bsim\b', r'\bactivation\b', r'\bekyc\b', r'\bnew sim\b'],
    'Mobile App Experience': [r'\bapp\b', r'\bmobile app\b', r'\bui\b', r'\binterface\b'],
    'OTT Bundle Services': [r'\bott\b', r'\bhotstar\b', r'\bnetflix\b', r'\bprime video\b', r'\bbundle\b'],
    'Pricing': [r'\bpric(e|ing)\b', r'\bcost\b', r'\bexpensive\b', r'\bcheap\b', r'\baffordable\b'],
    'Value for Money': [r'\bvalue for money\b', r'\bworth\b', r'\bvalue\b'],
    'Data Validity': [r'\bvalidity\b', r'\bexpir[ey]\b', r'\bvalid\b'],
    '5G Experience': [r'\b5g\b'],
    'Network Outage': [r'\boutage\b', r'\bmaintenance\b', r'\brestoration\b', r'\bdisruption\b', r'\bnetwork down\b'],
    'Number Portability': [r'\bportability\b', r'\bmnp\b', r'\bport(ed|ing)?\s+(my|the|a)?\s*number\b', r'\bnumber port\b'],
    'SMS Services': [r'\bsms\b', r'\botp\b', r'\btext message\b'],
    'Postpaid Plans': [r'\bpostpaid\b', r'\bmonthly plan\b'],
    'Network Congestion': [r'\bcongestion\b', r'\bpeak hour\b', r'\brush hour\b', r'\bthrottle\b'],
    'International Calling': [r'\binternational call\b', r'\bisd\b', r'\bstd\b', r'\babroad call\b'],
    'Device Compatibility': [r'\bdevice\b', r'\bcompatib\b', r'\bhandset\b', r'\bsamsung\b', r'\biphone\b']
}

SENTIMENT_COLORS = {
    'Positive': '#10B981',
    'Neutral': '#F59E0B',
    'Negative': '#EF4444'
}


# --- NLP Setup ---
stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'never'}
MIN_TOKEN_LENGTH = 3  # Remove tokens ≤2 chars (covers slang like "fr", "u", "ur")
lemmatizer = WordNetLemmatizer()
vader_analyzer = SentimentIntensityAnalyzer()


# --- Model Loading ---
AVAILABLE_MODELS = {
    'Tuned Logistic Regression': {
        'file': 'models/best_sentiment_model.pkl',
        'description': 'Best overall model (F1: 84.4%)',
        'f1': '84.4%',
        'type': 'sklearn'
    },
    'Naive Bayes': {
        'file': 'models/naive_bayes_model.pkl',
        'description': 'Lightweight & fast (F1: 84.0%)',
        'f1': '84.0%',
        'type': 'sklearn'
    },
    'SGD-SVM': {
        'file': 'models/sgd_svm_model.pkl',
        'description': 'Linear SVM via SGD (F1: 82.6%)',
        'f1': '82.6%',
        'type': 'sklearn'
    },
    'DistilBERT': {
        'file': 'models/distilbert_sentiment',
        'description': 'Transformer model (F1: 96.1%)',
        'f1': '96.1%',
        'type': 'distilbert'
    }
}


def load_models(model_name='Tuned Logistic Regression'):
    """Load the specified model, vectorizer, and label encoder from disk.
    
    Args:
        model_name: One of 'Tuned Logistic Regression', 'Naive Bayes', 'SGD-SVM', 'DistilBERT'
    
    Returns:
        tuple: (model, vectorizer, label_encoder)
        For DistilBERT: vectorizer is the tokenizer, model is the transformer model.
    
    Raises:
        FileNotFoundError: If required model files are missing.
        ValueError: If model_name is not recognized.
    """
    import os

    model_info = AVAILABLE_MODELS.get(model_name)
    if model_info is None:
        logger.warning(f"Unknown model '{model_name}', falling back to Tuned Logistic Regression.")
        model_info = AVAILABLE_MODELS['Tuned Logistic Regression']

    model_path = model_info['file']

    # --- DistilBERT Loading ---
    if model_info.get('type') == 'distilbert':
        if not os.path.exists(model_path):
            logger.error(f"DistilBERT model directory not found: {model_path}")
            raise FileNotFoundError(f"DistilBERT model not found: {model_path}")
        
        try:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            import torch
            
            tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            bert_model = DistilBertForSequenceClassification.from_pretrained(model_path)
            bert_model.eval()
            
            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            bert_model = bert_model.to(device)
            
            label_encoder = joblib.load('models/sentiment_label_encoder.pkl')
            logger.info(f"Loaded DistilBERT model from {model_path} on {device}")
            
            # Return bert_model with tokenizer attached as attribute
            bert_model._tokenizer = tokenizer
            bert_model._device = device
            bert_model._model_type = 'distilbert'
            return bert_model, tokenizer, label_encoder
            
        except ImportError:
            logger.error("transformers/torch not installed. Install with: pip install transformers torch")
            raise ImportError("Install transformers and torch: pip install transformers torch")
    
    # --- Sklearn Model Loading ---
    if not os.path.exists(model_path):
        model_path = 'models/best_sentiment_model.pkl'
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    encoder_path = 'models/sentiment_label_encoder.pkl'

    if not os.path.exists(vectorizer_path):
        logger.error(f"Vectorizer not found: {vectorizer_path}")
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
    if not os.path.exists(encoder_path):
        logger.error(f"Label encoder not found: {encoder_path}")
        raise FileNotFoundError(f"Label encoder not found: {encoder_path}")

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        label_encoder = joblib.load(encoder_path)
        model._model_type = 'sklearn'
        logger.info(f"Loaded model: {model_name} from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model files: {e}")
        raise

    return model, vectorizer, label_encoder


def get_available_models():
    """Return list of models that actually exist on disk."""
    import os
    available = {}
    for name, info in AVAILABLE_MODELS.items():
        path = info['file']
        # DistilBERT is a directory, others are files
        if info.get('type') == 'distilbert':
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'config.json')):
                available[name] = info
        else:
            if os.path.exists(path):
                available[name] = info
    return available


# --- Preprocessing ---
def preprocess_text(text):
    """Apply the full preprocessing pipeline to raw text.
    
    Steps: lowercase → remove URLs/emails → remove special chars →
    tokenize → remove stopwords (keep negation) → remove short tokens → lemmatize
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) >= MIN_TOKEN_LENGTH]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


# --- Aspect Identification ---
def identify_aspects(text):
    """Identify aspects mentioned in feedback using keyword regex matching."""
    text_lower = text.lower()
    identified = []
    for aspect, patterns in ASPECT_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                identified.append(aspect)
                break
    return identified if identified else ['General']


def identify_aspects_ml(text):
    """Identify aspects using the trained ML classifier.
    
    Falls back to keyword matching if classifier models aren't available.
    Uses the ML classifier to catch implicit aspect mentions that keywords miss,
    then merges with keyword results for best coverage.
    
    Args:
        text: Raw feedback text string.
    
    Returns:
        list: Detected aspect names.
    """
    import os
    
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("Empty or invalid text passed to identify_aspects_ml.")
        return ['General']
    
    # Check if ML models exist
    if not all(os.path.exists(f) for f in [
        'models/aspect_classifier.pkl',
        'models/aspect_mlb.pkl', 
        'models/aspect_tfidf.pkl'
    ]):
        return identify_aspects(text)
    
    # Load ML models (cached after first load)
    if not hasattr(identify_aspects_ml, '_models'):
        try:
            identify_aspects_ml._models = {
                'classifier': joblib.load('models/aspect_classifier.pkl'),
                'mlb': joblib.load('models/aspect_mlb.pkl'),
                'tfidf': joblib.load('models/aspect_tfidf.pkl')
            }
            logger.info("ML aspect classifier loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ML aspect models: {e}. Falling back to keywords.")
            return identify_aspects(text)
    
    models = identify_aspects_ml._models
    
    # Get keyword-based aspects
    keyword_aspects = identify_aspects(text)
    
    # Get ML-based aspects
    try:
        processed = preprocess_text(text)
        X = models['tfidf'].transform([processed])
        proba = models['classifier'].predict_proba(X)[0]
        
        # Get aspects with probability > 0.4
        ml_aspects = []
        for idx, p in enumerate(proba):
            if p > 0.4:
                ml_aspects.append(models['mlb'].classes_[idx])
    except Exception as e:
        logger.error(f"ML aspect prediction failed: {e}. Using keyword results.")
        return keyword_aspects
    
    # Merge: use keyword results if they found something specific,
    # otherwise use ML results. If keywords only found 'General', prefer ML.
    if keyword_aspects == ['General'] and ml_aspects:
        return ml_aspects
    elif keyword_aspects != ['General']:
        # Merge both, ML can add aspects keywords missed
        combined = list(dict.fromkeys(keyword_aspects + ml_aspects))
        return combined
    else:
        return keyword_aspects


# --- Clause Splitting ---
def split_into_clauses(text):
    """Split text into clauses on sentence boundaries AND contrastive conjunctions.
    
    Handles: but, however, though, although, yet, while, whereas, nevertheless, nonetheless
    """
    sentences = sent_tokenize(text)
    clauses = []
    clause_splitters = r'\b(but|however|though|although|yet|while|whereas|nevertheless|nonetheless|on the other hand)\b'
    for sent in sentences:
        parts = re.split(clause_splitters, sent, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if part and part.lower() not in ('but', 'however', 'though', 'although', 'yet', 'while', 'whereas', 'nevertheless', 'nonetheless', 'on the other hand'):
                if len(part.split()) >= 3:
                    clauses.append(part)
    return clauses if clauses else [text]


def get_relevant_sentences(raw_text, aspect):
    """Extract clauses/sentences that mention the given aspect.
    
    If only one clause matches and the text has contrastive structure,
    includes the adjacent clause to preserve full sentiment context.
    """
    clauses = split_into_clauses(raw_text)
    patterns = ASPECT_KEYWORDS.get(aspect, [])
    relevant = []
    relevant_indices = []
    
    for idx, clause in enumerate(clauses):
        clause_lower = clause.lower()
        for pattern in patterns:
            if re.search(pattern, clause_lower):
                relevant.append(clause)
                relevant_indices.append(idx)
                break
    
    # If we found the aspect in one clause but the text has contrastive structure,
    # include the adjacent clause (it likely refers to the same topic implicitly)
    if len(relevant) == 1 and len(clauses) > 1:
        idx = relevant_indices[0]
        if idx + 1 < len(clauses):
            next_clause = clauses[idx + 1]
            # Check if next clause mentions a DIFFERENT aspect explicitly
            matches_other_aspect = False
            for other_aspect, other_patterns in ASPECT_KEYWORDS.items():
                if other_aspect == aspect:
                    continue
                for p in other_patterns:
                    if re.search(p, next_clause.lower()):
                        matches_other_aspect = True
                        break
                if matches_other_aspect:
                    break
            # Include it if it doesn't belong to another aspect
            if not matches_other_aspect:
                relevant.append(next_clause)
    
    return ' '.join(relevant) if relevant else raw_text


# --- VADER ---
def get_vader_sentiment(text):
    """Get VADER compound sentiment score for raw text."""
    scores = vader_analyzer.polarity_scores(text)
    return scores['compound']


# --- Domain-Aware Sentiment Lexicon ---
# These phrases/words carry clear sentiment in telecom domain
# but VADER scores them as 0 (neutral)
DOMAIN_POSITIVE_PHRASES = [
    r'\bcompleted\b.*\b(within|quickly|smoothly|fast|hour|minute)',
    r'\b(seamless|seamlessly)\b', r'\b(hassle.?free)\b',
    r'\b(no issues?|no problem|no delay|without.{0,15}delay)',
    r'\b(works? perfectly|worked perfectly)\b',
    r'\b(smooth|smoothly)\b', r'\b(intuitive)\b',
    r'\b(convenient|conveniently)\b', r'\b(guided)\b',
    r'\b(reliable|reliably)\b', r'\b(stable|stability)\b',
    r'\b(impressed|impressive)\b', r'\b(resolved quickly)\b',
    r'\b(excellent|superb|fantastic|amazing|blazing)\b',
    r'\b(perfectly|professionally)\b',
    r'\b(generous|sufficient)\b', r'\b(competitive)\b',
]

DOMAIN_NEGATIVE_PHRASES = [
    r'\btoo (long|slow|short|expensive|high|many|much|frequent)\b',
    r'\b(takes? forever|took forever|takes? too long)\b',
    r'\b(took\s+\w+\s+days?)\b',  # "took three days", "took 5 days"
    r'\b(runs? out|ran out)\b.*\b(quickly|fast|too)\b',
    r'\b(hours? late|days? late|delayed|delays?)\b',
    r'\b(not compatible|incompatible)\b',
    r'\b(higher than|more than|overpriced|overcharged)\b',
    r'\b(never (resolved|fixed|solved|delivered|arrived|worked|responded))\b',
    r'\b(kept transferring|never solved|no solution|unresolved)\b',
    r'\b(crashes?|crashing|crash)\b', r'\b(freezes?|freezing)\b',
    r'\b(rejected|failed|fails?)\b',
    r'\b(unhelpful|useless|hopeless|pathetic|terrible|horrible|awful|worst)\b',
    r'\b(disappointing|disappointed)\b',
    r'\b(frequent outages?|multiple outages?)\b',
    r'\b(no signal|no coverage|no service|network down)\b',
    r'\b(unreliable|unstable)\b',
    r'\b(confusing|confused|incorrect|errors?)\b',
    r'\b(much higher|significantly higher)\b',
    r'\b(very little|too little|hardly any)\b',
    r'\b(not (worth|justified|justify))\b',
    r'\b(waste of)\b',
]


def get_domain_sentiment(text):
    """Get domain-specific sentiment signal from telecom phrases.
    
    Returns: 'positive', 'negative', or 'neutral'
    """
    text_lower = text.lower()
    
    pos_matches = sum(1 for p in DOMAIN_POSITIVE_PHRASES if re.search(p, text_lower))
    neg_matches = sum(1 for p in DOMAIN_NEGATIVE_PHRASES if re.search(p, text_lower))
    
    if pos_matches > 0 and neg_matches == 0:
        return 'positive'
    elif neg_matches > 0 and pos_matches == 0:
        return 'negative'
    elif pos_matches > 0 and neg_matches > 0:
        return 'mixed'
    return 'neutral'


# --- DistilBERT Prediction ---
def predict_absa_bert(raw_text, model, tokenizer):
    """ABSA prediction using DistilBERT.
    
    DistilBERT uses raw text (no preprocessing needed) with aspect prefix.
    """
    import torch
    
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
        return []

    aspects = identify_aspects_ml(raw_text)
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    device = model._device

    results = []
    for aspect in aspects:
        try:
            relevant_text = get_relevant_sentences(raw_text, aspect)
            # BERT input: [Aspect Name] raw text (no preprocessing)
            bert_input = f"[{aspect}] {relevant_text}"
            
            encoding = tokenizer(
                bert_input,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                pred = int(np.argmax(probs))
                confidence = float(probs[pred])
            
            results.append({
                'aspect': aspect,
                'sentiment': sentiment_map[pred],
                'confidence': confidence
            })
        except Exception as e:
            logger.error(f"DistilBERT prediction failed for aspect '{aspect}': {e}")
            results.append({
                'aspect': aspect,
                'sentiment': 'Neutral',
                'confidence': 0.0
            })

    return results


# --- Prediction Pipeline ---
def predict_absa(raw_text, model, vectorizer):
    """Full ABSA pipeline: identify aspects → isolate clauses → predict sentiment.
    
    Automatically routes to DistilBERT or sklearn pipeline based on model type.
    
    For sklearn models: Uses ML-based aspect identification, TF-IDF vectorization,
    and a layered correction system (VADER + domain lexicon).
    
    For DistilBERT: Uses raw text with aspect prefix, no preprocessing needed.
    
    Args:
        raw_text: Raw customer feedback string.
        model: Trained sentiment model (sklearn or DistilBERT).
        vectorizer: Fitted TF-IDF vectorizer (sklearn) or tokenizer (DistilBERT).
    
    Returns:
        list[dict]: List of {'aspect': str, 'sentiment': str, 'confidence': float}
    """
    if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
        logger.warning("Empty or invalid input to predict_absa.")
        return []

    # Route to DistilBERT pipeline if applicable
    if hasattr(model, '_model_type') and model._model_type == 'distilbert':
        return predict_absa_bert(raw_text, model, vectorizer)

    aspects = identify_aspects_ml(raw_text)
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    results = []
    for aspect in aspects:
        try:
            relevant_text = get_relevant_sentences(raw_text, aspect)
            processed = preprocess_text(relevant_text)
            absa_input = aspect.lower().replace(' ', '_') + ' ' + processed
            X = vectorizer.transform([absa_input])

            pred = model.predict(X)[0]
            confidence = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))

            # Get correction signals
            vader_score = get_vader_sentiment(relevant_text)
            domain_signal = get_domain_sentiment(relevant_text)

            # --- Correction Layer 1: VADER (general sentiment words) ---
            if confidence is not None and confidence < 0.60:
                if vader_score <= -0.3 and pred != 0:
                    pred = 0
                    confidence = max(confidence, abs(vader_score))
                elif vader_score >= 0.3 and pred != 2:
                    pred = 2
                    confidence = max(confidence, abs(vader_score))

            # --- Correction Layer 2: Domain lexicon (telecom-specific) ---
            if confidence is not None and confidence < 0.65 and pred == 1:
                if domain_signal == 'positive':
                    pred = 2
                    confidence = max(confidence, 0.55)
                elif domain_signal == 'negative':
                    pred = 0
                    confidence = max(confidence, 0.55)

            # --- Correction Layer 3: Strong domain override ---
            if pred == 1 and confidence is not None and confidence < 0.75:
                if domain_signal == 'negative':
                    pred = 0
                    confidence = max(confidence, 0.55)
                elif domain_signal == 'positive' and vader_score >= -0.1:
                    pred = 2
                    confidence = max(confidence, 0.55)

            # --- Correction Layer 4: Domain override for wrong Positive ---
            if pred == 2 and confidence is not None and confidence < 0.60:
                if domain_signal == 'negative':
                    pred = 0
                    confidence = max(confidence, 0.55)

            results.append({
                'aspect': aspect,
                'sentiment': sentiment_map[pred],
                'confidence': confidence
            })

        except Exception as e:
            logger.error(f"Prediction failed for aspect '{aspect}': {e}")
            results.append({
                'aspect': aspect,
                'sentiment': 'Neutral',
                'confidence': 0.0
            })

    logger.debug(f"Predicted {len(results)} aspects for: '{raw_text[:50]}...'")
    return results


# --- Helper Functions ---
def get_sentiment_score(results):
    """Compute a 0-100 overall sentiment score from results."""
    if not results:
        return 50
    score_map = {'Positive': 100, 'Neutral': 50, 'Negative': 0}
    return int(np.mean([score_map[r['sentiment']] for r in results]))


def get_overall_sentiment(results):
    """Determine the overall sentiment label (Positive/Negative/Neutral/Mixed)."""
    if not results:
        return "N/A"
    sentiments = [r['sentiment'] for r in results]
    from collections import Counter
    counts = Counter(sentiments)
    if counts.get('Positive', 0) > 0 and counts.get('Negative', 0) > 0:
        return "Mixed"
    return counts.most_common(1)[0][0]
