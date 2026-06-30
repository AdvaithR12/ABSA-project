"""
ABSA Model Utilities
---
Orchestration layer: model loading, prediction pipeline routing,
and helper functions used by the Streamlit app.

Core logic (preprocessing, aspect detection, sentiment) lives in the src/ package.
This module provides the public API consumed by app.py, evaluate_model.py, etc.
"""

import os
import logging
import numpy as np
import joblib
import nltk

# --- Import shared logic from src/ package ---
from src.preprocessing import preprocess_text
from src.aspect_detection import (
    ASPECT_KEYWORDS,
    identify_aspects_keyword as identify_aspects,
    identify_aspects_ml,
    split_into_clauses,
    get_relevant_sentences,
)
from src.sentiment import (
    DOMAIN_POSITIVE_PHRASES,
    DOMAIN_NEGATIVE_PHRASES,
    get_vader_score as get_vader_sentiment,
    get_domain_sentiment,
    predict_sentiment,
)
from src.logger import setup_logger

# --- Logging Setup ---
logger = setup_logger('absa.model_utils', log_file='logs/absa_inference.log')

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)


# --- Constants ---
SENTIMENT_COLORS = {
    'Positive': '#10B981',
    'Neutral': '#F59E0B',
    'Negative': '#EF4444'
}


# --- Model Loading ---
AVAILABLE_MODELS = {
    'Tuned Logistic Regression': {
        'file': 'models/best_sentiment_model.pkl',
        'description': 'Best overall model (F1: 84.9%)',
        'f1': '84.9%',
        'type': 'sklearn'
    },
    'Naive Bayes': {
        'file': 'models/naive_bayes_model.pkl',
        'description': 'Lightweight & fast (F1: 84.6%)',
        'f1': '84.6%',
        'type': 'sklearn'
    },
    'SGD-SVM': {
        'file': 'models/sgd_svm_model.pkl',
        'description': 'Linear SVM via SGD (F1: 85.3%)',
        'f1': '85.3%',
        'type': 'sklearn'
    },
    'DistilBERT': {
        'file': 'models/distilbert_sentiment',
        'description': 'Transformer model (F1: 95.6%)',
        'f1': '95.6%',
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
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            bert_model = bert_model.to(device)
            
            label_encoder = joblib.load('models/sentiment_label_encoder.pkl')
            logger.info(f"Loaded DistilBERT model from {model_path} on {device}")
            
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
    available = {}
    for name, info in AVAILABLE_MODELS.items():
        path = info['file']
        if info.get('type') == 'distilbert':
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'config.json')):
                available[name] = info
        else:
            if os.path.exists(path):
                available[name] = info
    return available


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
    
    For sklearn models: Uses ML-based aspect identification, clause isolation,
    and the layered correction system (VADER + domain lexicon) via src.sentiment.
    
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
    results = []

    for aspect in aspects:
        try:
            relevant_text = get_relevant_sentences(raw_text, aspect)
            result = predict_sentiment(relevant_text, aspect, model, vectorizer)
            results.append(result)
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
