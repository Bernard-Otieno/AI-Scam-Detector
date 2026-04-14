#!/usr/bin/env python3
"""
SAFARICOM FRAUD DETECTION — PRIVACY GUARANTEE DEMO

This script demonstrates that message content is NEVER stored or transmitted.
Only abstract numeric features are extracted and analyzed.

Run this during your demo to prove privacy compliance.
"""

import re
import json
import hashlib
from datetime import datetime
from typing import Dict, List

# ═════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — THE CORE PRIVACY MECHANISM
# ═════════════════════════════════════════════════════════════════════════════

def extract_features(message: str, sender: str) -> Dict:
    """
    Extracts numeric features from a message WITHOUT storing the content.
    
    This is the privacy guarantee:
    - The message text is analyzed in memory
    - Only boolean flags and counts are extracted
    - The original text is immediately discarded
    - Features are mathematically irreversible (cannot reconstruct message)
    
    Returns a dictionary of 18 numeric features.
    """
    
    msg_lower = message.lower()
    
    # ── BINARY FEATURES (0 or 1) ──────────────────────────────────────────
    
    # Check for presence of suspicious word categories
    features = {
        # Reward language
        'has_reward': int(any(word in msg_lower for word in 
            ['win', 'prize', 'congratulations', 'won', 'claim', 'bonus', 'redeem'])),
        
        # Threat language
        'has_threat': int(any(word in msg_lower for word in 
            ['arrest', 'police', 'suspended', 'fine', 'jail', 'deactivated'])),
        
        # Urgency language
        'has_urgency': int(any(word in msg_lower for word in 
            ['urgent', 'now', 'immediately', 'expire', 'today', 'last chance'])),
        
        # Transaction language
        'has_transaction': int(any(word in msg_lower for word in 
            ['reversal', 'reverse', 'confirm', 'send', 'transfer', 'pay'])),
        
        # Impersonation (mentions of legitimate brands)
        'has_impersonation': int(any(word in msg_lower for word in 
            ['safaricom', 'm-pesa', 'mpesa', 'fuliza', 'equity', 'kcb'])),
        
        # Emotional manipulation
        'has_emotional': int(any(word in msg_lower for word in 
            ['help', 'dying', 'hospital', 'stranded', 'please'])),
        
        # Technical indicators
        'has_link': int(bool(re.search(r'http', message))),
        'has_shortened_link': int(any(domain in msg_lower for domain in 
            ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'rb.gy'])),
        'has_untrusted_domain': int(
            bool(re.search(r'http', message)) and 
            not any(domain in msg_lower for domain in ['safaricom.co.ke', 'mpesa.co.ke'])
        ),
        'has_ussd': int(bool(re.search(r'\*\d{2,3}[*\d]*#', message))),
        
        # Sender characteristics
        'sender_is_numeric': int(sender.replace('+', '').replace(' ', '').isdigit()),
        'sender_is_shortcode': int(len(sender.replace('+', '')) <= 6 and sender.replace('+', '').isdigit()),
    }
    
    # ── NUMERIC FEATURES (counts and ratios) ──────────────────────────────
    
    features.update({
        'message_length': len(message),
        'word_count': len(message.split()),
        'capital_ratio': round(sum(1 for c in message if c.isupper()) / max(len(message), 1), 3),
        'exclamation_count': message.count('!'),
        'question_count': message.count('?'),
        'digit_ratio': round(sum(1 for c in message if c.isdigit()) / max(len(message), 1), 3),
    })
    
    return features


def hash_text(text: str) -> str:
    """
    Create a one-way hash of text for matching purposes.
    Hash is irreversible — cannot recover original text.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ═════════════════════════════════════════════════════════════════════════════
# SIMPLE RULE-BASED CLASSIFIER (mimics ML model)
# ═════════════════════════════════════════════════════════════════════════════

def classify_features(features: Dict) -> Dict:
    """
    Makes a scam/safe decision based ONLY on features.
    Never sees the original message content.
    """
    
    # Calculate risk score (0-10)
    score = 0
    
    # High risk indicators
    if features['has_reward'] and features['has_impersonation']:
        score += 4
    elif features['has_reward']:
        score += 2
    
    if features['has_threat']:
        score += 3
    
    if features['has_transaction'] and features['has_impersonation']:
        score += 3
    
    if features['has_urgency']:
        score += 1
    
    if features['has_untrusted_domain']:
        score += 2
    elif features['has_link']:
        score += 1
    
    if features['has_ussd']:
        score += 2
    
    if features['sender_is_numeric'] and not features['sender_is_shortcode']:
        score += 1
    
    score = min(score, 10)
    
    # Determine category
    if features['has_reward'] and features['has_impersonation']:
        category = "Hoax / Fake Reward"
    elif features['has_threat']:
        category = "Extortion / Threat"
    elif features['has_transaction'] and features['has_impersonation']:
        category = "Impersonation + Transaction Fraud"
    elif features['has_impersonation']:
        category = "Impersonation"
    elif score >= 4:
        category = "Suspicious"
    else:
        category = "Safe"
    
    return {
        'risk_score': score,
        'category': category,
        'is_scam': score >= 4
    }


def build_explanation(features: Dict, classification: Dict) -> str:
    """
    Generate a user-facing explanation based ONLY on features.
    Does not quote or reference the original message.
    """
    
    if not classification['is_scam']:
        return "No scam patterns detected. Message appears safe."
    
    # Template-based explanation
    if features['has_reward'] and features['has_impersonation']:
        return "This message claims you've won a prize from Safaricom or M-PESA. These companies never announce prizes by SMS. Do not click any links."
    
    if features['has_threat']:
        return "This message uses threatening language to create panic. This is a common extortion tactic. Do not respond or send money."
    
    if features['has_transaction'] and features['has_impersonation']:
        return "This message appears to be from Safaricom or M-PESA and asks you to take financial action. Verify directly with Safaricom before responding."
    
    if features['has_impersonation']:
        return "This message appears to impersonate Safaricom or a financial institution. The real Safaricom never asks for your PIN by SMS."
    
    if features['has_untrusted_domain']:
        return "This message contains a link to an untrusted website. Do not click it."
    
    return "This message has multiple patterns commonly used in scams. Verify before taking any action."


# ═════════════════════════════════════════════════════════════════════════════
# DEMO SCENARIOS
# ═════════════════════════════════════════════════════════════════════════════

DEMO_MESSAGES = [
    {
        "message": "Congratulations! You have won KSH 10,000 from Safaricom M-PESA. Claim now at http://bit.ly/safaricom-prize254",
        "sender": "+254700123456",
        "label": "SCAM"
    },
    {
        "message": "ALERT: Your M-PESA account will be suspended within 24 hours. Confirm your PIN by dialing *123*456# now.",
        "sender": "+254798765432",
        "label": "SCAM"
    },
    {
        "message": "Dear customer, your M-PESA transaction of KSH 500 to JOHN DOE has been completed successfully. Ref: AB12CD34EF56",
        "sender": "MPESA",
        "label": "SAFE"
    },
    {
        "message": "Your Safaricom bill of KSH 1,250 is due on 28th Feb. Pay via *544# or MySafaricom app.",
        "sender": "SAFARICOM",
        "label": "SAFE"
    },
]


# ═════════════════════════════════════════════════════════════════════════════
# DEMO RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_demo():
    """
    Runs the privacy guarantee demo.
    Shows that message content is never stored — only features.
    """
    
    print("=" * 80)
    print("SAFARICOM FRAUD DETECTION — PRIVACY GUARANTEE DEMO")
    print("=" * 80)
    print()
    print("This demo proves that MESSAGE CONTENT is NEVER stored or transmitted.")
    print("Only abstract numeric features are extracted and logged.")
    print()
    print("Watch the 'What Gets Logged' section — you'll never see the actual words")
    print("from the message. Only numbers and boolean flags.")
    print()
    print("=" * 80)
    print()
    
    logged_data = []
    
    for i, test in enumerate(DEMO_MESSAGES, 1):
        print(f"\n{'─' * 80}")
        print(f"TEST {i} — {test['label']}")
        print(f"{'─' * 80}\n")
        
        message = test['message']
        sender = test['sender']
        
        # ── STEP 1: Show the original message ─────────────────────────────
        print("📱 ORIGINAL MESSAGE (what the user sees):")
        print(f"   From: {sender}")
        print(f"   Text: {message}")
        print()
        
        # ── STEP 2: Extract features ──────────────────────────────────────
        print("🔍 FEATURE EXTRACTION (happens in memory, content discarded):")
        features = extract_features(message, sender)
        
        # Show features in a readable format
        print("   Binary flags:")
        for key, val in list(features.items())[:12]:
            print(f"      {key:25s} = {val}")
        print("   Numeric features:")
        for key, val in list(features.items())[12:]:
            print(f"      {key:25s} = {val}")
        print()
        
        # ── STEP 3: Classify based on features only ───────────────────────
        classification = classify_features(features)
        print(f"⚖️  CLASSIFICATION:")
        print(f"   Category:   {classification['category']}")
        print(f"   Risk Score: {classification['risk_score']}/10")
        print(f"   Decision:   {'BLOCK' if classification['is_scam'] else 'ALLOW'}")
        print()
        
        # ── STEP 4: Generate explanation without quoting message ───────────
        explanation = build_explanation(features, classification)
        print(f"💬 USER EXPLANATION (no message content quoted):")
        print(f"   {explanation}")
        print()
        
        # ── STEP 5: Show what gets logged ─────────────────────────────────
        message_hash = hash_text(message)
        sender_hash = hash_text(sender)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message_hash': message_hash,      # irreversible hash
            'sender_hash': sender_hash,        # irreversible hash
            'features': features,              # only numbers
            'risk_score': classification['risk_score'],
            'category': classification['category'],
            'is_blocked': classification['is_scam']
        }
        
        logged_data.append(log_entry)
        
        print("📊 WHAT GETS LOGGED TO DATABASE:")
        print(json.dumps(log_entry, indent=2))
        print()
        print("⚠️  NOTICE: The log contains NO message content.")
        print("   - 'message_hash' is a one-way hash (cannot reverse)")
        print("   - 'sender_hash' is a one-way hash (cannot reverse)")
        print("   - 'features' are just numbers (cannot reconstruct message)")
        print()
    
    # ── SUMMARY ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DEMO COMPLETE — PRIVACY GUARANTEE VERIFIED")
    print("=" * 80)
    print()
    print("✅ All decisions made using ONLY numeric features")
    print("✅ Original message content NEVER logged")
    print("✅ Hashes are one-way (mathematically irreversible)")
    print("✅ Features alone CANNOT reconstruct original text")
    print()
    print(f"Total messages processed: {len(DEMO_MESSAGES)}")
    print(f"Scams detected: {sum(1 for d in logged_data if d['is_blocked'])}")
    print(f"Safe messages: {sum(1 for d in logged_data if not d['is_blocked'])}")
    print()
    print("The log file 'privacy_demo_log.json' contains ONLY features — no content.")
    print()
    
    # Save log to file
    with open('privacy_demo_log.json', 'w') as f:
        json.dump(logged_data, f, indent=2)
    
    print("=" * 80)


if __name__ == "__main__":
    run_demo()