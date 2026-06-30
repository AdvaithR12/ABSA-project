"""
Text preprocessing module for the ABSA system.
Handles all text cleaning, tokenization, normalization, and spell correction.
"""

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.spell_correction import correct_spelling

# NLP resources
_stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'never'}
_lemmatizer = WordNetLemmatizer()

# Minimum token length — removes slang filler like "fr", "u", "ur"
# Matches the training notebook's MIN_TOKEN_LENGTH = 3
MIN_TOKEN_LENGTH = 3


def preprocess_text(text):
    """Apply the full preprocessing pipeline to raw text.
    
    Pipeline: lowercase → remove URLs/emails → remove special chars →
    spell correction → tokenize → remove stopwords (keep negation) →
    remove short tokens (< 3 chars) → lemmatize
    
    Args:
        text: Raw input text string.
    
    Returns:
        str: Cleaned and normalized text.
    """
    if not text or not isinstance(text, str):
        return ''

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Spell correction before tokenization
    text = correct_spelling(text)

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in _stop_words and len(w) >= MIN_TOKEN_LENGTH]
    tokens = [_lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)
