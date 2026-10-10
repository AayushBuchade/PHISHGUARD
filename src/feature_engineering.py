"""
PhishGuard - Cybersecurity Feature Engineering Module
Extracts domain-specific threat indicators and structural metrics from raw emails.
Transforms domain knowledge into quantitative features for tabular ML & threat scoring.
"""

import re
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Targeted high-frequency phishing keywords by threat category
URGENCY_KEYWORDS = [
    r'\burgent\b', r'\bimmediate(?:ly)?\b', r'\baction required\b', r'\battention\b',
    r'\bsuspend(?:ed|ing)?\b', r'\bterminate(?:d)?\b', r'\bexpir(?:e|ed|ing|ation)\b',
    r'\bwithin 24 hours\b', r'\bwithin 48 hours\b', r'\blimited time\b', r'\bcritical\b',
    r'\bunauthorized\b', r'\bsecurity alert\b', r'\block(?:ed)?\b', r'\brestrict(?:ed)?\b'
]

CREDENTIAL_KEYWORDS = [
    r'\bpassword\b', r'\bpasscode\b', r'\bcredentials?\b', r'\blogin\b', r'\bsign in\b',
    r'\bverify (?:your )?(?:account|identity|details)\b', r'\bupdate (?:your )?(?:account|profile|info)\b',
    r'\bconfirm (?:your )?(?:password|identity)\b', r'\bssn\b', r'\bsocial security\b',
    r'\bsecurity code\b', r'\bpin\b', r'\botp\b', r'\bauthentication\b', r'\btoken\b'
]

FINANCIAL_KEYWORDS = [
    r'\binvoice\b', r'\bpayment\b', r'\bwire transfer\b', r'\bbank\b', r'\brefund\b',
    r'\btransaction\b', r'\bbilling\b', r'\bcredit card\b', r'\bdebit card\b',
    r'\bwire\b', r'\bbitcoin\b', r'\bcrypto\b', r'\bethereum\b', r'\bwallet\b',
    r'\bdollars?\b', r'\bcompensation\b', r'\bbeneficiary\b', r'\blottery\b'
]

BRAND_TARGETS = [
    r'\bpaypal\b', r'\bmicrosoft\b', r'\boffice365\b', r'\bgoogle\b', r'\bapple\b',
    r'\bamazon\b', r'\bnetflix\b', r'\bwellsfargo\b', r'\bchase\b', r'\bbank of america\b',
    r'\bciti(?:bank)?\b', r'\bdhl\b', r'\bfedex\b', r'\bups\b', r'\birs\b'
]

SUSPICIOUS_TLDS = [
    r'\.xyz\b', r'\.top\b', r'\.tk\b', r'\.pw\b', r'\.buzz\b', r'\.ru\b',
    r'\.cn\b', r'\.club\b', r'\.work\b', r'\.support\b', r'\.live\b', r'\.stream\b'
]

URL_PATTERN = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+', re.IGNORECASE)
IP_IN_URL_PATTERN = re.compile(r'https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/[^\s<>"]*)?', re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')


def count_regex_matches(text: str, patterns: List[str]) -> int:
    """Calculates sum of occurrences matching an array of regex patterns."""
    count = 0
    lower_text = text.lower()
    for pat in patterns:
        count += len(re.findall(pat, lower_text))
    return count


def extract_cyber_features(email_text: str) -> Dict[str, Any]:
    """
    Extracts structured cybersecurity and linguistic features from a single raw email.
    
    Returns a dictionary of features:
    - url_count: Total URLs found
    - ip_url_count: URLs containing raw IPv4 address
    - urgency_count: Urgency and coercion cue count
    - credential_count: Credential and password harvesting keywords
    - financial_count: Financial transaction and payment keywords
    - brand_mention_count: Impersonation target brand mentions
    - suspicious_tld_count: Count of links pointing to high-risk TLDs
    - html_tag_count: Count of raw HTML markup tags
    - capital_ratio: Ratio of uppercase characters to total alpha characters
    - exclamation_count: Total exclamation mark count
    - question_count: Total question mark count
    - char_count: Total character length
    - word_count: Total word count
    """
    if not isinstance(email_text, str) or not email_text.strip():
        return {
            "url_count": 0,
            "ip_url_count": 0,
            "urgency_count": 0,
            "credential_count": 0,
            "financial_count": 0,
            "brand_mention_count": 0,
            "suspicious_tld_count": 0,
            "html_tag_count": 0,
            "capital_ratio": 0.0,
            "exclamation_count": 0,
            "question_count": 0,
            "char_count": 0,
            "word_count": 0,
        }

    urls = URL_PATTERN.findall(email_text)
    ip_urls = IP_IN_URL_PATTERN.findall(email_text)
    html_tags = HTML_TAG_PATTERN.findall(email_text)
    
    alpha_chars = [c for c in email_text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    capital_ratio = len(upper_chars) / len(alpha_chars) if alpha_chars else 0.0
    
    words = email_text.split()
    
    features = {
        "url_count": len(urls),
        "ip_url_count": len(ip_urls),
        "urgency_count": count_regex_matches(email_text, URGENCY_KEYWORDS),
        "credential_count": count_regex_matches(email_text, CREDENTIAL_KEYWORDS),
        "financial_count": count_regex_matches(email_text, FINANCIAL_KEYWORDS),
        "brand_mention_count": count_regex_matches(email_text, BRAND_TARGETS),
        "suspicious_tld_count": count_regex_matches(email_text, SUSPICIOUS_TLDS),
        "html_tag_count": len(html_tags),
        "capital_ratio": round(capital_ratio, 4),
        "exclamation_count": email_text.count("!"),
        "question_count": email_text.count("?"),
        "char_count": len(email_text),
        "word_count": len(words),
    }
    return features


def extract_features_df(series: pd.Series) -> pd.DataFrame:
    """
    Transforms a Pandas Series of email texts into a DataFrame of cybersecurity features.
    """
    features_list = [extract_cyber_features(text) for text in series]
    return pd.DataFrame(features_list)


if __name__ == "__main__":
    sample_phish = """
    URGENT NOTICE! We detected unauthorized login to your PayPal account from IP 192.168.1.1.
    Your account is suspended. Verify credentials immediately at http://192.168.1.1/paypal-login.xyz
    Failure to verify within 24 hours will result in permanent termination!
    """
    sample_safe = """
    Hi team, please find attached the quarterly product roadmap deck for tomorrow's review meeting.
    Let me know if you have any questions or feedback before 3 PM.
    Thanks,
    Alex
    """
    print("Phishing email features:")
    print(extract_cyber_features(sample_phish))
    print("\nLegitimate email features:")
    print(extract_cyber_features(sample_safe))
