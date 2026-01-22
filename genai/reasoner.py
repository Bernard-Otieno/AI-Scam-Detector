# genai/reasoner.py
import json
import requests
from config import OLLAMA_MODEL, OLLAMA_URL

def genai_classify(message: str, sender: str, rule_flags: list[str]) -> dict:
    prompt = f"""You are FraudGuard — real-time scam detector for Safaricom.
Message: "{message}"
Sender: {sender}
Quick flags: {', '.join(rule_flags) if rule_flags else 'none'}  # Now includes ML prob if passed

Detect: urgency, threats, rewards, impersonation (Safaricom/M-PESA/Fuliza), fake reversals, links, USSD misuse, emotional manipulation (fear/reward/urgency/isolation).
Classify ONLY as: Safe | Hoax | Extortion | Impersonation | Transactional Scam | High Risk
Risk score: 0–10 (integer)
One-sentence explanation.

Output format (exactly):
Category: X
Risk: Y
Explanation: Z"""
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.15}  # Low for consistency/speed
    }
    
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=6)
        resp.raise_for_status()
        result = json.loads(resp.text)
        output = result.get("response", "").strip()
        
        lines = [l.strip() for l in output.splitlines() if ":" in l]
        parsed = {}
        for line in lines:
            if line.startswith("Category:"): parsed["category"] = line.split(":", 1)[1].strip()
            if line.startswith("Risk:"): parsed["risk"] = int(line.split(":", 1)[1].strip())
            if line.startswith("Explanation:"): parsed["explanation"] = line.split(":", 1)[1].strip()
        
        if "category" not in parsed:
            raise ValueError("Parse failed")
        
        return parsed
    
    except Exception as e:
        return {
            "category": "High Risk",
            "risk": 0,
            "explanation": f"GenAI error: {str(e)}. Fallback to rules."
        }