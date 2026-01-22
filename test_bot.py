import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import json

# Synthetic data generator (from data/train_data.py)
def generate_synthetic_data(num_samples=100):  # Reduced for quick test
    scam_examples = [
        "Your M-PESA suspended! Dial *334# to reverse or lose funds.",
        "You won KSh 500,000 prize! Claim at bit.ly/claimnow.",
        "Safaricom alert: Account compromised. Send PIN to 254700123456.",
        "Emergency: Family in accident. Transfer KSh 10,000 urgently."
    ]
    safe_examples = [
        "Hi Bernard, meeting at 4 PM in Nairobi?",
        "Your order confirmed. Delivery tomorrow.",
        "You have 5 entries for Safaricom@25 Promo! Use M-PESA, Bonga, Buy bundles for a chance to win 1M! Dial *444*25#, *544*25#, *555*25# to Check Entries or OptOut",
        "Shangwe@25: Earn entries with M-PESA transactions. Check via *444*25#.",
        "Official Safaricom promo: Buy bundles to win prizes at @25 celebration."
    ]
    
    messages = []
    labels = []
    for _ in range(num_samples // 2):
        messages.extend(scam_examples)
        labels.extend([1] * len(scam_examples))
        messages.extend(safe_examples)
        labels.extend([0] * len(safe_examples))
    
    return messages[:num_samples], np.array(labels[:num_samples])

TRUSTED_SENDERS = ["SAFARICOM25", "SAFARICOM", "MPESA"]  # Known officials
PROMO_KEYWORDS = ["@25", "promo", "entries", "shangwe"]  # Legit promo signals

# Rule-based functions (from rules/rules.py)
def get_rule_score(message: str, sender: str) -> tuple:
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
    sender_lower = sender.lower()
    flags = []
    score = 0
    
    for cat, kws in keywords.items():
        if any(kw in msg_lower for kw in kws):
            score += 2
            flags.append(cat)
    
    if ussd_pattern.search(message):
        score += 3
        flags.append("ussd")
    
    links = link_pattern.findall(message)
    if links:
        score += 2
        flags.append("link")
    
    if sender.isdigit() or sender.startswith("254"):
        score += 1
        flags.append("suspicious_sender")
    if any(t in sender_lower for t in TRUSTED_SENDERS):
        flags = [f for f in flags if f != "impersonation"]
        score -= 2
    
    if any(p in msg_lower for p in PROMO_KEYWORDS):
        score -= 3
        flags.append("promo_legit")

    
    return score, flags, links

def extract_features(message: str, sender: str) -> np.ndarray:
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
    
    for kws in keywords.values():
        features.append(sum(1 for kw in kws if kw in msg_lower))
    
    features.append(1 if ussd_pattern.search(message) else 0)
    features.append(1 if link_pattern.search(message) else 0)
    features.append(1 if sender.isdigit() or sender.startswith("254") else 0)
    features.append(len(message) / 100.0)
    features.append(1 if any(p in msg_lower for p in PROMO_KEYWORDS) else 0)
    
    return np.array(features).reshape(1, -1)

# Simple ML Classifier (from ml/scorer.py)
class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.fc(x))

# Train ML model (reduced for test)
def train_ml_model():
    messages, labels = generate_synthetic_data(100)
    input_size = extract_features(messages[0], "dummy")[0].shape[0]
    
    model = SimpleClassifier(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for epoch in range(10):  # Reduced epochs
        for msg, label in zip(messages, labels):
            features = torch.tensor(extract_features(msg, "2547xxxxxx"), dtype=torch.float32)
            output = model(features)
            loss = criterion(output, torch.tensor([[label]], dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model

ML_MODEL = train_ml_model()

# Mock GenAI (simulates genai/reasoner.py output)
def mock_genai_classify(message: str, sender: str, extended_flags: list):
    if "transaction" in extended_flags or "ussd" in extended_flags:
        category = "Transactional Scam"
        risk = 7
        explanation = "Detected fake reversal or transaction pressure."
    elif "reward" in extended_flags:
        category = "Hoax"
        risk = 6
        explanation = "Detected prize win or reward scam."
    elif "threat" in extended_flags or "emotional" in extended_flags:
        category = "Extortion"
        risk = 8
        explanation = "Detected threats or emotional manipulation."
    elif "impersonation" in extended_flags:
        category = "Impersonation"
        risk = 7
        explanation = "Detected impersonation of trusted entity."
    else:
        category = "Safe"
        risk = 1
        explanation = "No suspicious patterns detected."
    
    return {"category": category, "risk": risk, "explanation": explanation}

# Hybrid scorer (from ml/scorer.py, no server)
def hybrid_score(message: str, sender: str = "Unknown") -> dict:
    rule_score, rule_flags, links = get_rule_score(message, sender)
    
    if "promo_legit" in rule_flags:
        rule_score = max(rule_score - 4, 0)

    features = torch.tensor(extract_features(message, sender), dtype=torch.float32)
    with torch.no_grad():
        ml_prob = ML_MODEL(features).item()
    ml_score = int(ml_prob * 10)
    
    extended_flags = rule_flags + [f"ML scam prob: {ml_prob:.2f}"]
    genai_result = mock_genai_classify(message, sender, extended_flags)
    
    final_risk = int((rule_score * 0.2 + ml_score * 0.3 + genai_result["risk"] * 0.5))
    if "promo_legit" in rule_flags:
        final_risk = max(final_risk - 2, 0)

    category = genai_result["category"]
    explanation = genai_result["explanation"] + f" (ML contrib: {ml_prob:.2f})"
    
    if "impersonation" in rule_flags:
        category = "Impersonation"
    elif "transaction" in rule_flags:
        category = "Transactional Scam"
    
    action = "BLOCK + Notify" if final_risk >= 4 else "ALLOW"
    
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S EAT"),
        "message": message,
        "sender": sender,
        "category": category,
        "risk": final_risk,
        "explanation": explanation,
        "action": action
    }
    
    return result

# Run tests
test_messages = [
    {"message": "Dear customer, your M-PESA account is suspended. Dial *123# now to reverse or lose funds!", "sender": "254700000000"},
    {"message": "You have won a prize! Claim now at bit.ly/winprize", "sender": "Unknown"},
    {"message": "Urgent: Send money to avoid arrest for unpaid fine.", "sender": "254712345678"},
    {"message": "Safaricom notice: Update your details or account will be closed.", "sender": "Safaricom"},
    {"message": "Hi, let's meet at 2 PM?", "sender": "Friend"},
    {"message": "You have 5 entries  for Safaricom@25 Promo!\nUse M-PESA, Bonga, Buy bundles for a chance to win 1M! Dial *444*25#, *544*25#, *555*25# to Check Entries or OptOut", "sender": "SAFARICOM25"},
    {"message": "Happy Customer Service Week! Thank you for banking with us. We value you and remain your listening, caring partner walking with you to power your dreams.", "sender": "EQUITY BANK"},
    {"message": "Please note that 8936 points in your Safaricom account expire today. We recommend redeeming them as soon as possible to avoid loss. https://safaricom-co.hair/ke", "sender": "0743941435"},
]

results = []
for msg in test_messages:
    result = hybrid_score(msg["message"], msg["sender"])
    results.append(result)

print(json.dumps(results, indent=2))











