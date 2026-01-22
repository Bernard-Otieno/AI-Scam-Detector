# ml/scorer.py
from datetime import datetime
import time
import requests
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rules.rules import get_rule_score, extract_features
from genai.reasoner import genai_classify
from config import SERVER_URL, CONSENT_GIVEN
from data.train_data import generate_synthetic_data

class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)  # Binary output (scam prob)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.fc(x))

def train_ml_model():
    messages, labels = generate_synthetic_data(500)  # More samples for better training
    input_size = extract_features(messages[0], "dummy")[0].shape[0]  # Get feature dim
    
    model = SimpleClassifier(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for epoch in range(50):  # Quick training
        for msg, label in zip(messages, labels):
            features = torch.tensor(extract_features(msg, "2547xxxxxx"), dtype=torch.float32)
            output = model(features)
            loss = criterion(output, torch.tensor([[label]], dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model

# Global ML model (train once on init)
ML_MODEL = train_ml_model()

def hybrid_score(message: str, sender: str = "Unknown") -> dict:
    start = time.time()
    
    # Layer 1: Rules (fast signals)
    rule_score, rule_flags, links = get_rule_score(message, sender)
    
    # Layer 2: Traditional ML (probabilistic score on features)
    features = torch.tensor(extract_features(message, sender), dtype=torch.float32)
    with torch.no_grad():
        ml_prob = ML_MODEL(features).item()  # Scam probability [0-1]
    ml_score = int(ml_prob * 10)  # Scale to 0-10 for consistency
    
    # Layer 3: GenAI (reasoning with rules + ML input)
    genai_result = genai_classify(message, sender, rule_flags + [f"ML scam prob: {ml_prob:.2f}"])
    
    # Final hybrid: Weighted average risk, GenAI category/explanation
    final_risk = int((rule_score * 0.2 + ml_score * 0.3 + genai_result.get("risk", 0) * 0.5))
    category = genai_result["category"]
    explanation = genai_result["explanation"] + f" (ML contrib: scam prob {ml_prob:.2f})"
    
    # Overrides from PDF signals
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
        "action": action,
        "latency_sec": round(time.time() - start, 2)
    }
    
    # Anonymized report to server (add ML prob for learning)
    if action.startswith("BLOCK") and CONSENT_GIVEN:
        report = {
            "type": category,
            "sender_hash": hash(sender),
            "links": [hash(link) for link in links],
            "flags": rule_flags,
            "ml_prob": ml_prob,
            "timestamp": result["timestamp"]
        }
        try:
            requests.post(f"{SERVER_URL}/report", json=report, timeout=2)
        except:
            print("Server report failed.")
    
    return result