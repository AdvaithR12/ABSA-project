"""
Spell correction module for the ABSA system.
Uses SymSpell for fast approximate string matching to handle
typos and misspellings in customer feedback.

Preserves telecom-specific terms that SymSpell might "correct" wrongly.
"""

import os
import re
import pkg_resources
from symspellpy import SymSpell, Verbosity

# --- Singleton SymSpell instance ---
_sym_spell = None

# Telecom domain terms to NEVER correct (would be mangled by generic dictionary)
PROTECTED_TERMS = {
    # Telecom-specific
    'volte', 'ekyc', 'ott', 'hotstar', 'netflix', 'sim', 'sms', 'otp',
    'isd', 'std', 'ivr', 'mnp', 'mbps', 'gb', 'tb', '4g', '5g', '3g', '2g',
    'wifi', 'topup', 'prepaid', 'postpaid', 'recharge', 'roaming',
    # Brand names that might get corrected
    'jio', 'airtel', 'vodafone', 'bsnl', 'vi',
    # Common abbreviations in reviews
    'app', 'ui', 'ux', 'hd', 'avg',
}

# Domain-specific corrections that SymSpell might get wrong
# Maps common misspellings to correct telecom terms
DOMAIN_CORRECTIONS = {
    'signl': 'signal',
    'signa': 'signal',
    'singal': 'signal',
    'singl': 'signal',
    'netwok': 'network',
    'ntwrk': 'network',
    'newtork': 'network',
    'internrt': 'internet',
    'intenet': 'internet',
    'intrnet': 'internet',
    'covrage': 'coverage',
    'coverge': 'coverage',
    'coveage': 'coverage',
    'cal': 'call',
    'cll': 'call',
    'calll': 'call',
    'billig': 'billing',
    'biling': 'billing',
    'billng': 'billing',
    'rechage': 'recharge',
    'recharg': 'recharge',
    'rechrge': 'recharge',
    'spedd': 'speed',
    'speeed': 'speed',
    'spped': 'speed',
    'downlod': 'download',
    'donwload': 'download',
    'uplod': 'upload',
    'bandwith': 'bandwidth',
    'bandwidht': 'bandwidth',
    'buffring': 'buffering',
    'bufering': 'buffering',
    'roaming': 'roaming',
    'romig': 'roaming',
    'suport': 'support',
    'supprt': 'support',
    'supoort': 'support',
    'custmer': 'customer',
    'cutomer': 'customer',
    'cusomer': 'customer',
    'complant': 'complaint',
    'complait': 'complaint',
}


def _init_symspell():
    """Initialize SymSpell with the built-in English frequency dictionary."""
    global _sym_spell
    if _sym_spell is not None:
        return _sym_spell

    _sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

    # Load the built-in frequency dictionary
    dict_path = pkg_resources.resource_filename(
        'symspellpy', 'frequency_dictionary_en_82_765.txt'
    )
    _sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

    return _sym_spell


def correct_spelling(text):
    """Correct spelling mistakes in text while preserving domain terms.

    Strategy:
    - Splits text into words
    - First checks domain-specific corrections (telecom typos)
    - Skips protected telecom terms, numbers, and very short words (<=2 chars)
    - Corrects each word using SymSpell (edit distance ≤ 2)
    - Preserves original word if no good suggestion found

    Args:
        text: Input text string (should already be lowercased).

    Returns:
        str: Text with spelling corrections applied.
    """
    if not text or not isinstance(text, str):
        return text

    sym_spell = _init_symspell()
    words = text.split()
    corrected_words = []

    for word in words:
        # Check domain-specific corrections first
        if word in DOMAIN_CORRECTIONS:
            corrected_words.append(DOMAIN_CORRECTIONS[word])
            continue

        # Skip protected terms, numbers, very short words
        if (word in PROTECTED_TERMS or
                len(word) <= 2 or
                re.match(r'^\d+$', word) or
                re.match(r'^\d+\s*(gb|mb|mbps|tb)$', word)):
            corrected_words.append(word)
            continue

        # Look up suggestion with max edit distance 2
        suggestions = sym_spell.lookup(
            word, Verbosity.CLOSEST, max_edit_distance=2
        )

        if suggestions and suggestions[0].distance <= 2:
            best = suggestions[0]
            if best.distance == 0:
                # Word is already correct
                corrected_words.append(word)
            elif best.count > 1000:
                # Only accept corrections from common words
                corrected_words.append(best.term)
            else:
                # Low-frequency suggestion — keep original to be safe
                corrected_words.append(word)
        else:
            # No suggestion found — keep original
            corrected_words.append(word)

    return ' '.join(corrected_words)
