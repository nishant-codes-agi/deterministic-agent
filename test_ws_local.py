import sys
import logging
logging.basicConfig(level=logging.ERROR)

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

print("Starting websocket connection test...")
try:
    with client.websocket_connect("/api/v1/runs/run-1e18fb67/stream") as websocket:
        print("Connected!")
        data = websocket.receive_text()
        print("Received:", data)
except Exception as e:
    import traceback
    traceback.print_exc()

