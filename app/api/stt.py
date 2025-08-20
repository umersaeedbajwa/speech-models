import os
from typing import Literal
from fastapi import APIRouter, Security, WebSocket
from app.auth import get_api_key_ws
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
import numpy as np
import librosa
from dotenv import load_dotenv
from functools import cache
import json

load_dotenv()

STT_REPO = os.getenv("STT_REPO")

router = APIRouter()

@cache
def load_moonshine(
    model_name: Literal["moonshine/base", "moonshine/tiny"],
) -> MoonshineOnnxModel:
    return MoonshineOnnxModel(model_name=model_name)

load_moonshine(STT_REPO)
tokenizer = load_tokenizer()

def audio_to_float32(audio_np: np.ndarray) -> np.ndarray:
    """Convert int16 numpy audio to float32 in [-1, 1]."""
    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    return audio_np

@router.websocket("/")
async def stt_websocket(
    websocket: WebSocket,
    authorization: str = Security(get_api_key_ws)
):
    """
    STT WebSocket endpoint for jambonz integration.
    
    Input:
        - WebSocket connection with Authorization header
        - Receives:
            * JSON control messages (e.g., {"type": "start", ...}, {"type": "stop"})
            * Binary audio frames (linear16 PCM, 8kHz)
    
    Output:
        - Sends JSON messages:
            * {"type": "transcription", "is_final": bool, "alternatives": [{"transcript": str, "confidence": float}], "language": str, "channel": int}
            * {"type": "error", "error": str}
        - Server is responsible for closing the WebSocket after stop message
    
    Integration:
        - jambonz will connect via WebSocket, send control/audio frames, and expect JSON responses
        - Your implementation should authenticate, process audio, and send transcriptions/errors as JSON
    """
    
    await websocket.accept()
    audio_bytes = b""
    language = "en-US"
    sample_rate = 8000
    moonshine = load_moonshine(STT_REPO)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # Accumulate audio bytes
                    audio_bytes += message["bytes"]
                elif "text" in message:
                    # Handle control messages (e.g., start/stop)
                    control = json.loads(message["text"])
                    print(f"Control message received: {control}")
                    if control.get("type") == "start":
                        language = control.get("language", "en-US")
                        sample_rate = control.get("sampleRateHz", 8000)
                        # You can handle other options here
                    elif control.get("type") == "stop":
                        print(f"Audio frame received: {len(audio_bytes)} bytes")
                        # On stop, process the accumulated audio
                        # Convert bytes to numpy array (int16)
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_np = audio_to_float32(audio_np)
                        if sample_rate != 16000:
                            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
                        if audio_np.ndim == 1:
                            audio_np = audio_np.reshape(1, -1)
                        tokens = moonshine.generate(audio_np)
                        transcription = tokenizer.decode_batch(tokens)[0].strip()
                        # Send result
                        await websocket.send_json({
                            "type": "transcription",
                            "is_final": True,
                            "alternatives": [{"transcript": transcription, "confidence": 1.0}],
                            "language": language,
                            "channel": 1
                        })
                        await websocket.close()
                        break
    except Exception as e:
        await websocket.send_json({"type": "error", "error": str(e)})
        await websocket.close()
