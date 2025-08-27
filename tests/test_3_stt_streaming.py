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
import numpy as np
import io
warnings.filterwarnings("ignore")

load_dotenv()

API_KEY = os.getenv("API_KEY", "testkey")
os.environ["PYTHONPATH"] = "."
client = TestClient(app)

def test_stt_websocket_streaming_with_pause():
    """Test STT with streaming audio chunks followed by silence to trigger pause detection"""
    
    # Find the latest .wav file in ./data to use as source audio
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    assert wav_files, "No .wav files found in ./data"
    latest_file = max(wav_files, key=os.path.getctime)
    
    # Read the audio file
    with sf.SoundFile(latest_file) as f:
        audio_data = f.read(dtype=np.float32)
        samplerate = f.samplerate
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert stereo to mono
    
    print(f"Testing streaming with file: {latest_file}")
    print(f"Audio length: {len(audio_data)} samples, Sample rate: {samplerate}")
    
    # Convert to int16 for transmission (simulating real audio stream)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Create silence buffer (1 second of silence)
    silence_duration = 1.0  # seconds
    silence_samples = int(samplerate * silence_duration)
    silence_int16 = np.zeros(silence_samples, dtype=np.int16)
    
    # Chunk size for streaming (simulate real-time audio chunks)
    chunk_duration = 0.1  # 100ms chunks
    chunk_size = int(samplerate * chunk_duration)
    
    with client.websocket_connect("/stt/", headers={"Authorization": f"Bearer {API_KEY}"}) as websocket:
        
        start_time = time.time()
        
        # Send start control message with streaming enabled
        websocket.send_json({
            "type": "start",
            "language": "en-US",
            "format": "raw",
            "encoding": "LINEAR16",
            "interimResults": False,
            "sampleRateHz": samplerate,
            "streaming": True,  # Enable streaming mode with ReplyOnPause
            "options": {}
        })
        
        print("Sending audio chunks...")
        
        # Send audio data in chunks
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size]
            websocket.send_bytes(chunk.tobytes())
            time.sleep(0.05)  # Small delay to simulate real-time streaming
        
        print("Sending silence to trigger pause detection...")
        
        # Send silence chunks to trigger pause detection
        for i in range(0, len(silence_int16), chunk_size):
            chunk = silence_int16[i:i + chunk_size]
            websocket.send_bytes(chunk.tobytes())
            time.sleep(0.05)  # Small delay
        
        print("Waiting for transcription response...")
        
        # Wait for transcription result (with timeout)
        response_received = False
        timeout = 30  # 30 seconds timeout
        start_wait = time.time()
        
        while not response_received and (time.time() - start_wait) < timeout:
            try:
                # Try to receive a message with a short timeout
                data = websocket.receive_json(mode="text")
                
                if data.get("type") == "transcription":
                    response_received = True
                    elapsed = time.time() - start_time
                    print(f"Time taken to get transcription: {elapsed:.2f} seconds")
                    print("STT Response:", data)
                    
                    # Verify the response structure
                    assert data.get("type") == "transcription"
                    assert "alternatives" in data
                    assert len(data["alternatives"]) > 0
                    assert "transcript" in data["alternatives"][0]
                    assert data.get("triggered_by") == "pause"
                    
                    # The transcript should contain recognizable words
                    transcript = data["alternatives"][0]["transcript"].lower()
                    assert len(transcript.strip()) > 0, "Transcript should not be empty"
                    
                    break
                elif data.get("type") == "error":
                    print("Error received:", data)
                    pytest.fail(f"STT error: {data.get('error')}")
                else:
                    print("Received other message:", data)
                    
            except Exception as e:
                # Continue waiting if no message available yet
                time.sleep(0.1)
                continue
        
        if not response_received:
            pytest.fail("No transcription response received within timeout period")
        
        # Send stop control message
        websocket.send_json({"type": "stop"})


def test_stt_websocket_streaming_only_silence():
    """Test that sending only silence doesn't trigger a transcription"""
    
    samplerate = 16000  # Standard sample rate
    
    # Create silence buffer (2 seconds of silence)
    silence_duration = 2.0  # seconds
    silence_samples = int(samplerate * silence_duration)
    silence_int16 = np.zeros(silence_samples, dtype=np.int16)
    
    # Chunk size for streaming
    chunk_duration = 0.1  # 100ms chunks
    chunk_size = int(samplerate * chunk_duration)
    
    with client.websocket_connect("/stt/", headers={"Authorization": f"Bearer {API_KEY}"}) as websocket:
        
        # Send start control message with streaming enabled
        websocket.send_json({
            "type": "start",
            "language": "en-US",
            "format": "raw",
            "encoding": "LINEAR16",
            "interimResults": False,
            "sampleRateHz": samplerate,
            "streaming": True,  # Enable streaming mode with ReplyOnPause
            "options": {}
        })
        
        print("Sending only silence chunks...")
        
        # Send only silence chunks
        for i in range(0, len(silence_int16), chunk_size):
            chunk = silence_int16[i:i + chunk_size]
            websocket.send_bytes(chunk.tobytes())
            time.sleep(0.02)  # Small delay
        
        print("Waiting to see if any transcription is triggered...")
        
        # Wait briefly to see if anything is transcribed
        time.sleep(2.0)
        
        # Try to receive a message - should not get a transcription
        transcription_received = False
        try:
            data = websocket.receive_json(mode="text")
            if data.get("type") == "transcription":
                transcription_received = True
                print("Unexpected transcription:", data)
        except:
            # No message received, which is expected
            pass
        
        # Should not have received a transcription for silence only
        assert not transcription_received, "Should not transcribe silence-only audio"
        
        # Send stop control message
        websocket.send_json({"type": "stop"})
        
        print("Test passed - no transcription for silence-only audio")
