"""
Aspect detection module for the ABSA system.
Uses fine-tuned DeBERTa model for multi-label aspect classification.

Falls back to TF-IDF + keyword validation if DeBERTa is unavailable.
"""

import os
import re
import json
import joblib
from src.preprocessing import preprocess_text
from src.logger import setup_logger

logger = setup_logger(__name__)

# Standard aspects for reference
STANDARD_ASPECTS = [
    'Network Coverage', 'Call Quality', 'Internet Speed', 'Network Outage',
    'Billing', 'Customer Support', 'Roaming', 'Postpaid Plans',
    'SIM Activation', 'Network Congestion', 'International Calling',
    '5G Experience', 'OTT Bundle Services', 'Data Balance',
    'Mobile App Experience', 'Pricing', 'Device Compatibility',
    'Recharge Plans', 'SMS Services', 'Number Portability',
    'Data Validity', 'Value for Money'
]

# Path to DeBERTa model
DEBERTA_MODEL_PATH = 'models/deberta_aspect'

# Cache for models
_deberta_model = None
_deberta_tokenizer = None
_deberta_config = None
_ml_models = None

# Default confidence threshold
DEFAULT_THRESHOLD = 0.5


# ============================================================
# DeBERTa-based Aspect Detection (Primary)
# ============================================================

def _load_deberta_model():
    """Load DeBERTa model and tokenizer (cached on first call)."""
    global _deberta_model, _deberta_tokenizer, _deberta_config
    
    if _deberta_model is not None:
        return _deberta_model, _deberta_tokenizer, _deberta_config
    
    if not os.path.exists(DEBERTA_MODEL_PATH):
        logger.warning(f"DeBERTa model not found at {DEBERTA_MODEL_PATH}")
        return None, None, None
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        logger.info(f"Loading DeBERTa model from {DEBERTA_MODEL_PATH}...")
        
        _deberta_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_PATH)
        _deberta_model = AutoModelForSequenceClassification.from_pretrained(
            DEBERTA_MODEL_PATH,
            torch_dtype=torch.float32
        )
        _deberta_model.eval()
        
        # Load label mapping
        label_mapping_path = os.path.join(DEBERTA_MODEL_PATH, 'label_mapping.json')
        with open(label_mapping_path) as f:
            _deberta_config = json.load(f)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _deberta_model = _deberta_model.to(device)
        _deberta_config['device'] = device
        
        logger.info(f"DeBERTa model loaded successfully on {device}")
        return _deberta_model, _deberta_tokenizer, _deberta_config
        
    except ImportError as e:
        logger.warning(f"transformers/torch not installed: {e}")
        return None, None, None
    except Exception as e:
        logger.error(f"Failed to load DeBERTa model: {e}")
        return None, None, None


def identify_aspects_deberta(text, threshold=DEFAULT_THRESHOLD):
    """Identify aspects using fine-tuned DeBERTa model.
    
    Args:
        text: Raw feedback text.
        threshold: Minimum confidence for aspect detection (default 0.5).
    
    Returns:
        list: Detected aspect names sorted by confidence, or ['General'] if none found.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return ['General']
    
    model, tokenizer, config = _load_deberta_model()
    if model is None:
        logger.warning("DeBERTa not available, falling back to TF-IDF")
        return identify_aspects_tfidf(text, threshold)
    
    import torch
    
    device = config['device']
    classes = config['classes']
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()
    
    # Get predictions above threshold
    predictions = [
        (classes[i], float(probs[i]))
        for i in range(len(probs)) if probs[i] >= threshold
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    if predictions:
        aspects = [aspect for aspect, _ in predictions]
        logger.debug(f"DeBERTa predictions: {predictions}")
        return aspects
    
    return ['General']


# ============================================================
# TF-IDF-based Aspect Detection (Fallback)
# ============================================================

# Validation keywords for TF-IDF fallback
ASPECT_VALIDATORS = {
    'Network Coverage': [r'\bcoverage\b', r'\bsignal\b', r'\bbars?\b', r'\bdead zone\b', r'\btower\b', r'\bnetwork\b', r'\breception\b'],
    'Internet Speed': [r'\binternet\b', r'\bspeed\b', r'\bmbps\b', r'\bdownload\b', r'\bupload\b', r'\bbandwidth\b', r'\bbuffering\b', r'\bstreaming\b', r'\bdata\b', r'\bwifi\b', r'\bbroadband\b', r'\b4g\b', r'\blte\b', r'\bfast\b', r'\bslow\b'],
    'Call Quality': [r'\bcall\b', r'\bvoice\b', r'\baudio\b', r'\bclarity\b', r'\bdrop\b', r'\becho\b', r'\bvolte\b', r'\bstatic\b', r'\bphone\b', r'\btalk\b', r'\bspeak\b'],
    'Customer Support': [r'\bsupport\b', r'\bcare\b', r'\bservice\b', r'\bhelpline\b', r'\bagent\b', r'\bcomplaint\b', r'\bticket\b', r'\bivr\b', r'\bexecutive\b', r'\brespond\b'],
    'Billing': [r'\bbill\b', r'\binvoice\b', r'\bcharge\b', r'\bpayment\b', r'\bpay\b', r'\bdebit\b', r'\brefund\b', r'\bcredit\b'],
    'Recharge Plans': [r'\brecharge\b', r'\bplan\b', r'\bpack\b', r'\bprepaid\b', r'\btopup\b', r'\bunlimited\b', r'\bcombo\b'],
    'Data Balance': [r'\bdata\b', r'\bbalance\b', r'\blimit\b', r'\busage\b', r'\bgb\b', r'\bquota\b', r'\bcap\b'],
    'Roaming': [r'\broaming\b', r'\btravel\b', r'\babroad\b', r'\binternational\b'],
    'SIM Activation': [r'\bsim\b', r'\bactivat\b', r'\bekyc\b', r'\bkyc\b', r'\bport\b'],
    'Mobile App Experience': [r'\bapp\b', r'\bmobile\b', r'\bui\b', r'\binterface\b', r'\bcrash\b', r'\blogin\b'],
    'OTT Bundle Services': [r'\bott\b', r'\bhotstar\b', r'\bnetflix\b', r'\bprime\b', r'\bbundle\b', r'\bsubscription\b'],
    'Pricing': [r'\bpric\b', r'\bcost\b', r'\bexpensive\b', r'\bcheap\b', r'\baffordable\b', r'\brate\b', r'\btariff\b'],
    'Value for Money': [r'\bvalue\b', r'\bworth\b', r'\bmoney\b'],
    'Data Validity': [r'\bvalidity\b', r'\bexpir\b', r'\bvalid\b', r'\bdays\b'],
    '5G Experience': [r'\b5g\b'],
    'Network Outage': [r'\boutage\b', r'\bmaintenance\b', r'\bdown\b', r'\bdisruption\b', r'\brestoration\b'],
    'Number Portability': [r'\bport\b', r'\bmnp\b', r'\bswitch\b', r'\btransfer\b'],
    'SMS Services': [r'\bsms\b', r'\botp\b', r'\btext\b', r'\bmessage\b'],
    'Postpaid Plans': [r'\bpostpaid\b', r'\bmonthly\b'],
    'Network Congestion': [r'\bcongestion\b', r'\bpeak\b', r'\bthrottle\b', r'\bslowed\b'],
    'International Calling': [r'\binternational\b', r'\bisd\b', r'\bstd\b', r'\boverseas\b', r'\babroad\b'],
    'Device Compatibility': [r'\bdevice\b', r'\bcompatib\b', r'\bhandset\b', r'\bphone\b', r'\bsamsung\b', r'\biphone\b', r'\bandroid\b']
}

TELECOM_INDICATORS = [
    r'\bnetwork\b', r'\bsignal\b', r'\bcoverage\b', r'\binternet\b', r'\bdata\b',
    r'\bspeed\b', r'\bcall\b', r'\bvoice\b', r'\bsim\b', r'\brecharge\b',
    r'\bplan\b', r'\bbill\b', r'\bpayment\b', r'\broaming\b', r'\b4g\b', r'\b5g\b',
    r'\blte\b', r'\bsms\b', r'\botp\b', r'\bapp\b', r'\bmobile\b', r'\bphone\b',
    r'\bprepaid\b', r'\bpostpaid\b', r'\bcarrier\b', r'\bprovider\b', r'\btelecom\b',
    r'\bairtel\b', r'\bjio\b', r'\bvi\b', r'\bbsnl\b', r'\bservice\b', r'\bsupport\b',
    r'\boutage\b', r'\btower\b', r'\bmbps\b', r'\bgb\b', r'\bdownload\b', r'\bupload\b'
]


def _load_ml_models():
    """Load TF-IDF ML models (cached on first call)."""
    global _ml_models
    if _ml_models is not None:
        return _ml_models

    required_files = [
        'models/aspect_classifier.pkl',
        'models/aspect_mlb.pkl',
        'models/aspect_tfidf.pkl'
    ]

    if not all(os.path.exists(f) for f in required_files):
        logger.warning("TF-IDF aspect classifier models not found.")
        return None

    _ml_models = {
        'classifier': joblib.load('models/aspect_classifier.pkl'),
        'mlb': joblib.load('models/aspect_mlb.pkl'),
        'tfidf': joblib.load('models/aspect_tfidf.pkl')
    }
    logger.info("TF-IDF aspect classifier loaded successfully.")
    return _ml_models


def _is_telecom_domain(text):
    """Check if text is related to telecom domain."""
    text_lower = text.lower()
    for pattern in TELECOM_INDICATORS:
        if re.search(pattern, text_lower):
            return True
    return False


def _validate_aspect(text, aspect):
    """Check if the text contains keywords relevant to the aspect."""
    text_lower = text.lower()
    validators = ASPECT_VALIDATORS.get(aspect, [])
    
    if not validators:
        return True
    
    for pattern in validators:
        if re.search(pattern, text_lower):
            return True
    return False


def identify_aspects_tfidf(text, threshold=DEFAULT_THRESHOLD):
    """Identify aspects using TF-IDF classifier with keyword validation.
    
    This is the fallback when DeBERTa is not available.
    
    Args:
        text: Raw feedback text.
        threshold: Minimum confidence for aspect detection.
    
    Returns:
        list: Detected aspect names or ['General'].
    """
    if not text or not isinstance(text, str) or not text.strip():
        return ['General']
    
    if not _is_telecom_domain(text):
        logger.debug(f"Text not in telecom domain: {text[:50]}...")
        return ['General']

    models = _load_ml_models()
    if models is None:
        return ['General']

    processed = preprocess_text(text)
    X = models['tfidf'].transform([processed])
    proba = models['classifier'].predict_proba(X)[0]
    
    candidates = [
        (models['mlb'].classes_[idx], prob)
        for idx, prob in enumerate(proba) if prob >= threshold
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    validated = []
    for aspect, prob in candidates:
        if _validate_aspect(text, aspect):
            validated.append((aspect, prob))
    
    if validated:
        return [aspect for aspect, _ in validated]
    
    return ['General']


# ============================================================
# Main Entry Points
# ============================================================

def identify_aspects_ml(text, threshold=DEFAULT_THRESHOLD):
    """Identify aspects using the best available model.
    
    Uses DeBERTa if available, falls back to TF-IDF.
    
    Args:
        text: Raw feedback text.
        threshold: Minimum confidence for aspect detection.
    
    Returns:
        list: Detected aspect names or ['General'].
    """
    return identify_aspects_deberta(text, threshold)


def identify_aspects(text, threshold=DEFAULT_THRESHOLD):
    """Main aspect identification function.
    
    Primary entry point for aspect detection.
    
    Args:
        text: Raw feedback text.
        threshold: Minimum confidence for aspect detection.
    
    Returns:
        list: Detected aspect names or ['General'].
    """
    return identify_aspects_deberta(text, threshold)


# Backward compatibility
identify_aspects_keyword = identify_aspects_ml


# ============================================================
# Clause Splitting (for sentiment extraction)
# ============================================================

from nltk.tokenize import sent_tokenize


def split_into_clauses(text):
    """Split text into clauses on sentence boundaries and contrastive conjunctions."""
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
    """Extract clauses/sentences relevant to the given aspect.
    
    Uses DeBERTa if available to score clause relevance.
    """
    clauses = split_into_clauses(raw_text)
    if len(clauses) <= 1:
        return raw_text
    
    model, tokenizer, config = _load_deberta_model()
    
    if model is not None:
        # Use DeBERTa for clause relevance
        import torch
        
        device = config['device']
        classes = config['classes']
        
        try:
            aspect_idx = classes.index(aspect)
        except ValueError:
            return raw_text
        
        relevant_clauses = []
        for clause in clauses:
            inputs = tokenizer(
                clause,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()
            
            if probs[aspect_idx] > 0.3:
                relevant_clauses.append(clause)
        
        return ' '.join(relevant_clauses) if relevant_clauses else raw_text
    
    else:
        # Fallback to TF-IDF
        models = _load_ml_models()
        if models is None:
            return raw_text
        
        try:
            aspect_idx = list(models['mlb'].classes_).index(aspect)
        except ValueError:
            return raw_text
        
        relevant_clauses = []
        for clause in clauses:
            processed = preprocess_text(clause)
            if not processed.strip():
                continue
            X = models['tfidf'].transform([processed])
            proba = models['classifier'].predict_proba(X)[0]
            
            if proba[aspect_idx] > 0.3:
                relevant_clauses.append(clause)
        
        return ' '.join(relevant_clauses) if relevant_clauses else raw_text


# For backward compatibility
ASPECT_KEYWORDS = {aspect: ASPECT_VALIDATORS.get(aspect, []) for aspect in STANDARD_ASPECTS}
