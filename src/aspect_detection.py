"""
Aspect detection module for the ABSA system.
Handles both keyword-based and ML-based aspect identification,
clause splitting, and aspect-relevant sentence extraction.
"""

import re
import os
import joblib
from nltk.tokenize import sent_tokenize
from src.preprocessing import preprocess_text
from src.spell_correction import correct_spelling
from src.logger import setup_logger

logger = setup_logger(__name__)

# Keyword patterns for each aspect (regex word-boundary matching)
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

# Cache for ML models (loaded once)
_ml_models = None


def identify_aspects_keyword(text):
    """Identify aspects using keyword regex matching.
    
    Applies spell correction before matching to handle typos.
    
    Args:
        text: Raw feedback text.
    
    Returns:
        list: Detected aspect names, or ['General'] if none found.
    """
    text_lower = text.lower()
    text_corrected = correct_spelling(text_lower)
    identified = []
    for aspect, patterns in ASPECT_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text_corrected):
                identified.append(aspect)
                break
    return identified if identified else ['General']


def _load_ml_models():
    """Load ML aspect classifier models (cached on first call)."""
    global _ml_models
    if _ml_models is not None:
        return _ml_models

    required_files = [
        'models/aspect_classifier.pkl',
        'models/aspect_mlb.pkl',
        'models/aspect_tfidf.pkl'
    ]

    if not all(os.path.exists(f) for f in required_files):
        logger.warning("ML aspect classifier models not found. Using keywords only.")
        return None

    _ml_models = {
        'classifier': joblib.load('models/aspect_classifier.pkl'),
        'mlb': joblib.load('models/aspect_mlb.pkl'),
        'tfidf': joblib.load('models/aspect_tfidf.pkl')
    }
    logger.info("ML aspect classifier loaded successfully.")
    return _ml_models


def identify_aspects_ml(text, threshold=0.4):
    """Identify aspects using ML classifier + keyword fallback.
    
    Strategy:
    - If keywords find specific aspects, merge with ML predictions
    - If keywords only find 'General', prefer ML predictions
    - Falls back to keyword-only if ML models unavailable
    
    Args:
        text: Raw feedback text.
        threshold: Minimum probability for ML aspect detection.
    
    Returns:
        list: Detected aspect names.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return ['General']

    models = _load_ml_models()
    if models is None:
        return identify_aspects_keyword(text)

    keyword_aspects = identify_aspects_keyword(text)

    # ML prediction
    processed = preprocess_text(text)
    X = models['tfidf'].transform([processed])
    proba = models['classifier'].predict_proba(X)[0]

    ml_aspects = [
        models['mlb'].classes_[idx]
        for idx, p in enumerate(proba) if p > threshold
    ]

    # Merge logic
    if keyword_aspects == ['General'] and ml_aspects:
        return ml_aspects
    elif keyword_aspects != ['General']:
        combined = list(dict.fromkeys(keyword_aspects + ml_aspects))
        return combined
    return keyword_aspects


# --- Clause Splitting ---
def split_into_clauses(text):
    """Split text into clauses on sentence boundaries AND contrastive conjunctions.
    
    Handles: but, however, though, although, yet, while, whereas, nevertheless, nonetheless
    
    Args:
        text: Raw feedback text.
    
    Returns:
        list: List of clause strings (minimum 3 words each).
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
    
    Args:
        raw_text: Full feedback text.
        aspect: Aspect name to look for.
    
    Returns:
        str: Concatenated relevant clauses, or full text if none match.
    """
    clauses = split_into_clauses(raw_text)
    patterns = ASPECT_KEYWORDS.get(aspect, [])
    relevant = []
    relevant_indices = []
    
    for idx, clause in enumerate(clauses):
        clause_lower = clause.lower()
        clause_corrected = correct_spelling(clause_lower)
        for pattern in patterns:
            if re.search(pattern, clause_corrected):
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
