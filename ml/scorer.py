# ml/scorer.py
from datetime import datetime
import time
import requests
import json
from rules.rules import get_rule_score
from genai.reasoner import genai_classify
from config import SERVER_URL, CONSENT_GIVEN

def hybrid_score(message: str, sender: str = "Unknown") -> dict:
    start = time.time()
    
    # Rule-based (fast, local)
    rule_score, rule_flags, links = get_rule_score(message, sender)
    
    # GenAI (local, reasoning layer)
    genai_result = genai_classify(message, sender, rule_flags)
    
    # Hybrid decision
    final_risk = max(rule_score, genai_result.get("risk", 0))
    category = genai_result["category"]
    explanation = genai_result["explanation"]
    
    # Override for strong signals (e.g., PDF impersonation/transaction focus)
    if "impersonation" in rule_flags:
        category = "Impersonation"
    elif "transaction" in rule_flags:
        category = "Transactional Scam"
    
    action = "BLOCK + Notify" if final_risk >= 4 else "ALLOW"
    
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S EAT"),
        "message": message,  # Not sent to server — local only
        "sender": sender,
        "category": category,
        "risk": final_risk,
        "explanation": explanation,
        "action": action,
        "latency_sec": round(time.time() - start, 2)
    }
    
    # Anonymized report (PDF section 6: type, sender, links, timestamp — no full message)
    if action.startswith("BLOCK") and CONSENT_GIVEN:
        report = {
            "type": category,
            "sender_hash": hash(sender),  # Anonymized
            "links": [hash(link) for link in links],  # Anonymized domains
            "flags": rule_flags,
            "timestamp": result["timestamp"]
        }
        try:
            requests.post(f"{SERVER_URL}/report", json=report, timeout=2)  # Async, non-blocking
        except:
            print("Server report failed — queued locally.")  # Fallback
    
    return result