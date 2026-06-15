"""
Text preprocessing module for the ABSA system.
Handles all text cleaning, tokenization, and normalization.
"""

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLP resources
_stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'never'}
_noise_words = {'lol', 'bruh', 'fr', 'u', 'ur', 'pls', 'plz'}
_lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Apply the full preprocessing pipeline to raw text.
    
    Pipeline: lowercase → remove URLs/emails → remove special chars →
    tokenize → remove stopwords (keep negation) → lemmatize → remove slang
    
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

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in _stop_words and w not in _noise_words]
    tokens = [_lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)
