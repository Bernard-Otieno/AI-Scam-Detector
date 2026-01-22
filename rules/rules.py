# rules/rules.py
import re
import numpy as np
from typing import Tuple, List
def get_rule_score(message: str, sender: str) -> tuple[int, list[str], list[str]]:
    keywords = {
        "urgency": ["urgent", "now", "immediate", "today", "expire", "suspend", "last chance"],
        "threat": ["threat", "arrest", "blackmail", "emergency", "fine", "jail", "police"],
        "reward": [
            "win", "prize", "claim", "bonus", "free", "congratulations",
            "points", "bonga", "redeem", "reward", "promo", "expire", "expires"
        ],
        "impersonation": ["safaricom", "m-pesa", "fuliza", "equity", "kcb", "government"],
        "transaction": ["reversal", "reverse", "confirm", "send", "transfer", "pay now"],
        "emotional": ["fear", "reward", "urgency", "isolation"]  # From PDF manipulation patterns
    }
    ussd_pattern = re.compile(r"\*\d{2,3}\#")
    link_pattern = re.compile(r"https?://\S+")
    
    msg_lower = message.lower()
    flags = []
    score = 0
    
    for cat, kws in keywords.items():
        if any(kw in msg_lower for kw in kws):
            score += 2
            flags.append(cat)
    
    if ussd_pattern.search(message):
        score += 3
        flags.append("ussd")
    
    trusted_domains = ["safaricom.co.ke", "https://bit.ly/mpesalnk", "https://www.baze.co.ke", "pesapal.com"]

    links = link_pattern.findall(message)
    if links:
        score += 2
        flags.append("link")

        for link in links:
            if not any(td in link for td in trusted_domains):
                score += 3
                flags.append("untrusted_domain")
    
    if sender.isdigit() or sender.startswith("254"):
        score += 1
        flags.append("suspicious_sender")
    
    return score, flags, links

def extract_features(message: str, sender: str) -> np.ndarray:
    # Simple features for traditional ML: keyword counts + binaries (shape: [num_features])
    keywords = {
        "urgency": ["urgent", "now", "immediate", "today", "expire", "suspend", "last chance"],
        "threat": ["threat", "arrest", "blackmail", "emergency", "fine", "jail", "police"],
        "reward": ["win", "prize", "claim", "bonus", "free money", "congratulations"],
        "impersonation": ["safaricom", "m-pesa", "fuliza", "equity", "kcb", "government"],
        "transaction": ["reversal", "reverse", "confirm", "send", "transfer", "pay now"],
        "emotional": ["fear", "reward", "urgency", "isolation"]
    }
    ussd_pattern = re.compile(r"\*\d{2,3}\#")
    link_pattern = re.compile(r"https?://\S+")
    
    msg_lower = message.lower()
    features = []
    
    # Count-based features
    for kws in keywords.values():
        features.append(sum(1 for kw in kws if kw in msg_lower))
    
    # Binary features
    features.append(1 if ussd_pattern.search(message) else 0)
    features.append(1 if link_pattern.search(message) else 0)
    features.append(1 if sender.isdigit() or sender.startswith("254") else 0)
    features.append(len(message) / 100.0)  # Normalized length
    
    return np.array(features).reshape(1, -1)  # For batch=1 inference
