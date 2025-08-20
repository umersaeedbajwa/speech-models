import time
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from fastapi import WebSocket
from app.main import app
import os
from dotenv import load_dotenv
import warnings
import glob
warnings.filterwarnings("ignore")

load_dotenv()

API_KEY = os.getenv("API_KEY", "testkey")

client = TestClient(app)

def test_stt_websocket_from_latest_wav():
    # Find the latest .wav file in ./data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    assert wav_files, "No .wav files found in ./data"
    latest_file = max(wav_files, key=os.path.getctime)

    # Read audio bytes
    with open(latest_file, "rb") as f:
        audio_bytes = f.read()

    with sf.SoundFile(latest_file) as f:
        samplerate = f.samplerate

    print("Testing file:", latest_file)

    with client.websocket_connect("/stt/", headers={"Authorization": f"Bearer {API_KEY}"}) as websocket:
        
        start_time = time.time()
        # Send start control message
        websocket.send_json({
            "type": "start",
            "language": "en-US",
            "format": "raw",
            "encoding": "LINEAR16",
            "interimResults": False,
            "sampleRateHz": samplerate,
            "options": {}
        })
        # Send audio bytes (simulate jambonz binary frame)
        websocket.send_bytes(audio_bytes)
        # Send stop control message
        websocket.send_json({"type": "stop"})

        # Receive transcription result
        data = websocket.receive_json()
        
        elapsed = time.time() - start_time
        print(f"Time taken to generate text: {elapsed:.2f} seconds")
        print("STT Response:", data)
        assert data.get("type") == "transcription"
        assert "alternatives" in data
        assert len(data["alternatives"]) > 0
        assert "transcript" in data["alternatives"][0]