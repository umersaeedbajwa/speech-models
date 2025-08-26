import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
from dotenv import load_dotenv
import soundfile as sf
import io
import datetime
import time
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

API_KEY = os.getenv("API_KEY", "testkey")
os.environ["PYTHONPATH"] = "."

client = TestClient(app)

def test_tts_post_success():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {
        "language": "en-US",
        "voice": "af_heart",
        "type": "text",
        "text": "Combined STT and TTS is not working yet with JamBonz."
    }
    print("Sending Request:", data)
    start_time = time.time()
    response = client.post("/tts/", json=data, headers=headers)
    elapsed = time.time() - start_time
    print(f"Time taken to generate voice: {elapsed:.2f} seconds")
    assert response.status_code == 200

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, f"test_output_{timestamp}.wav")
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Audio saved to {filename}")


    audio, samplerate = sf.read(io.BytesIO(response.content))
    print("Audio shape:", audio.shape, "Sample rate:", samplerate)
    


