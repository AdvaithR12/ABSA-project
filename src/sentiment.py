"""
Sentiment prediction module for the ABSA system.
Handles model inference, VADER correction, and domain lexicon overrides.
"""

import re
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from src.logger import setup_logger

logger = setup_logger(__name__)

# VADER analyzer (initialized once)
_vader = SentimentIntensityAnalyzer()

# --- Domain-Aware Sentiment Lexicon ---
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
    r'\b(took\s+\w+\s+days?)\b',
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


def get_vader_score(text):
    """Get VADER compound sentiment score.
    
    Args:
        text: Raw text to analyze.
    
    Returns:
        float: Compound score between -1.0 and 1.0.
    """
    return _vader.polarity_scores(text)['compound']


def get_domain_sentiment(text):
    """Get domain-specific sentiment signal from telecom phrases.
    
    Args:
        text: Raw text to analyze.
    
    Returns:
        str: 'positive', 'negative', 'mixed', or 'neutral'.
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


def predict_sentiment(text, aspect, model, vectorizer, thresholds=None):
    """Predict sentiment for a single aspect-text pair with layered corrections.
    
    Args:
        text: Relevant text for this aspect (after clause isolation).
        aspect: Aspect name string.
        model: Trained sklearn model with predict/predict_proba.
        vectorizer: Fitted TF-IDF vectorizer.
        thresholds: Dict of correction thresholds (from config). Uses defaults if None.
    
    Returns:
        dict: {'aspect': str, 'sentiment': str, 'confidence': float}
    """
    from src.preprocessing import preprocess_text

    # Default thresholds
    if thresholds is None:
        thresholds = {
            'vader_correction_threshold': 0.60,
            'vader_negative_score': -0.3,
            'vader_positive_score': 0.3,
            'domain_neutral_override_threshold': 0.65,
            'domain_strong_override_threshold': 0.75,
            'domain_positive_wrong_threshold': 0.60,
        }

    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    # Prepare model input
    processed = preprocess_text(text)
    absa_input = aspect.lower().replace(' ', '_') + ' ' + processed
    X = vectorizer.transform([absa_input])

    # Model prediction
    pred = model.predict(X)[0]
    confidence = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))

    # Correction signals
    vader_score = get_vader_score(text)
    domain_signal = get_domain_sentiment(text)

    # Layer 1: VADER correction
    if confidence is not None and confidence < thresholds['vader_correction_threshold']:
        if vader_score <= thresholds['vader_negative_score'] and pred != 0:
            pred = 0
            confidence = max(confidence, abs(vader_score))
        elif vader_score >= thresholds['vader_positive_score'] and pred != 2:
            pred = 2
            confidence = max(confidence, abs(vader_score))

    # Layer 2: Domain lexicon (Neutral override)
    if confidence is not None and confidence < thresholds['domain_neutral_override_threshold'] and pred == 1:
        if domain_signal == 'positive':
            pred = 2
            confidence = max(confidence, 0.55)
        elif domain_signal == 'negative':
            pred = 0
            confidence = max(confidence, 0.55)

    # Layer 3: Strong domain override
    if pred == 1 and confidence is not None and confidence < thresholds['domain_strong_override_threshold']:
        if domain_signal == 'negative':
            pred = 0
            confidence = max(confidence, 0.55)
        elif domain_signal == 'positive' and vader_score >= -0.1:
            pred = 2
            confidence = max(confidence, 0.55)

    # Layer 4: Wrong Positive override
    if pred == 2 and confidence is not None and confidence < thresholds['domain_positive_wrong_threshold']:
        if domain_signal == 'negative':
            pred = 0
            confidence = max(confidence, 0.55)

    return {
        'aspect': aspect,
        'sentiment': sentiment_map[pred],
        'confidence': confidence
    }
