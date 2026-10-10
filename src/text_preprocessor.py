"""
PhishGuard - Text Preprocessing Module
Implements cybersecurity-aware NLP text cleaning, normalization, and tokenization.
Preserves threat cues (urgency, tokenized URLs, IP addresses) while stripping noise.
"""

import re
import html
import string
from typing import List, Optional

# Security-sensitive words that must NOT be removed even if they appear in standard stopword lists
CYBER_PRESERVE_WORDS = {
    "urgent", "immediately", "now", "verify", "update", "account", "login", 
    "password", "security", "alert", "bank", "action", "confirm", "risk", 
    "suspended", "unauthorized", "click", "here", "access", "expire", "warning"
}

# Standard English stopwords (concise set to avoid external NLTK download dependencies in runtime)
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", 
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", 
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here's", 
    "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", 
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", 
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", 
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", 
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", 
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", 
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", 
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", 
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", 
    "you've", "your", "yours", "yourself", "yourselves"
}

# Final stopword set filtered against cybersecurity preserve words
STOPWORDS = STANDARD_STOPWORDS - CYBER_PRESERVE_WORDS

# Regex patterns for normalization
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+', re.IGNORECASE)
IP_REGEX = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
CURRENCY_REGEX = re.compile(r'[\$\£\€]\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+\s*(?:dollars|usd|euros|btc|bitcoin)\b', re.IGNORECASE)
HTML_TAG_REGEX = re.compile(r'<[^>]+>')
EXCESS_PUNCT_REGEX = re.compile(r'([!?.]){2,}')
WHITESPACE_REGEX = re.compile(r'\s+')


def clean_text(raw_text: Optional[str], preserve_tokens: bool = True) -> str:
    """
    Cleans raw email text for machine learning.
    
    Steps:
    1. Null handling & HTML unescaping
    2. HTML tag stripping
    3. Tokenizing URLs, IP addresses, emails, currencies into special domain tokens
    4. Punctuation normalization
    5. Case folding & whitespace compaction
    
    Args:
        raw_text: Raw email body or subject string.
        preserve_tokens: Whether to replace URLs/IPs with explicit threat tokens.
    
    Returns:
        Cleaned, normalized string ready for vectorization.
    """
    if raw_text is None or not isinstance(raw_text, str):
        return ""
    
    # Unescape HTML entities (&amp;, &lt;, &gt;, &nbsp;)
    text = html.unescape(raw_text)
    
    # Strip HTML tags
    text = HTML_TAG_REGEX.sub(" ", text)
    
    if preserve_tokens:
        # Mask IP addresses first before general numbers
        text = IP_REGEX.sub(" token_ip_address ", text)
        # Mask URLs
        text = URL_REGEX.sub(" token_url_link ", text)
        # Mask email addresses
        text = EMAIL_REGEX.sub(" token_email_address ", text)
        # Mask monetary figures
        text = CURRENCY_REGEX.sub(" token_currency_amount ", text)
    
    # Normalize multiple exclamation / question marks (e.g. "!!!" -> " ! ")
    text = EXCESS_PUNCT_REGEX.sub(r' \1 ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove characters outside standard alphanumeric, underscores and punctuation
    text = re.sub(r'[^a-z0-9\s_!?.$,-]', ' ', text)
    
    # Collapse consecutive whitespace
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    return text


def tokenize_and_filter_stopwords(text: str) -> List[str]:
    """
    Tokenizes text into words and removes common stopwords while retaining cyber-threat keywords.
    """
    tokens = re.findall(r'\b[a-z0-9_]{2,}\b', text.lower())
    filtered = [token for token in tokens if token not in STOPWORDS]
    return filtered


def preprocess_for_tfidf(raw_text: Optional[str]) -> str:
    """
    End-to-end string preprocessor for Scikit-Learn TF-IDF vectorizer.
    Returns cleaned text with stopwords filtered, maintaining token order.
    """
    cleaned = clean_text(raw_text, preserve_tokens=True)
    tokens = tokenize_and_filter_stopwords(cleaned)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = """
    URGENT SECURITY ALERT! Your Bank of America account has been suspended due to unauthorized access from IP 192.168.1.105.
    Please verify your credentials immediately at http://secure-boa-update.com/login?id=88234 or your account will be deleted within 24 hours!
    Payment required: $500.00.
    """
    print("--- RAW TEXT ---")
    print(sample.strip())
    print("\n--- CLEANED TEXT ---")
    print(clean_text(sample))
    print("\n--- PREPROCESSED FOR TF-IDF ---")
    print(preprocess_for_tfidf(sample))
