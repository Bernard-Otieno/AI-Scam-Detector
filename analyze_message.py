# analyze_message.py (On-device client)
import json
from ml.scorer import hybrid_score

if __name__ == "__main__":
    print("FraudGuard On-Device Scanner (real-time). Type 'exit' to quit.\n")
    
    while True:
        msg = input("Incoming message: ").strip()
        if msg.lower() == "exit":
            break
        sender = input("Sender: ").strip() or "2547xxxxxx"
        
        result = hybrid_score(msg, sender)
        print("\n" + json.dumps(result, indent=2))
        if result["action"].startswith("BLOCK"):
            print("Notification: Blocked scam — " + result["explanation"])
        print("=" * 70 + "\n")