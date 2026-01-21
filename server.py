# server.py (Server-side: Run with `flask run`)
from flask import Flask, request, jsonify
import json
from datetime import datetime

app = Flask(__name__)

# Simulated aggregated intelligence (e.g., for learning/blacklisting)
reports = []  # In prod: DB like PostgreSQL on Railway
blacklist_senders = set()
emerging_trends = {}  # e.g., {"Hoax": 5, "Impersonation": 3}

@app.route('/report', methods=['POST'])
def receive_report():
    report = request.json
    reports.append(report)
    
    # Simulate learning: Update trends/blacklists
    category = report["type"]
    emerging_trends[category] = emerging_trends.get(category, 0) + 1
    blacklist_senders.add(report["sender_hash"])  # Accelerate blacklisting
    
    print(f"Received report: {report}")  # Log for demo
    return jsonify({"status": "received", "trends": emerging_trends}), 200

@app.route('/updates', methods=['GET'])
def get_updates():
    # Push model updates (e.g., new keywords/prompt tweaks based on trends)
    updates = {
        "new_keywords": {"urgency": ["act fast"]},  # Simulated evolution
        "blacklist": list(blacklist_senders),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S EAT")
    }
    return jsonify(updates), 200

if __name__ == "__main__":
    app.run(debug=True)