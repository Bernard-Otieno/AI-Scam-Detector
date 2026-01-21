# config.py
OLLAMA_MODEL = "gemma3:1b"  # Fast local SLM for on-device (change to phi4:mini if slower)
OLLAMA_URL = "http://localhost:11434/api/generate"
SERVER_URL = "http://127.0.0.1:5000"  # Local dev; replace with Railway URL in prod
CONSENT_GIVEN = True  # Simulated one-time consent for reporting