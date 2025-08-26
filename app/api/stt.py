import os
from typing import Literal
from fastapi import APIRouter, Security, WebSocket
from app.auth import get_api_key_ws
import numpy as np
from dotenv import load_dotenv
import json
from app.service.stt import get_stt_model

load_dotenv()


STT_REPO = os.getenv("STT_REPO")
STT_PAUSE = int(os.getenv("STT_PAUSE", 1500))  # Ensure integer

router = APIRouter()


@router.websocket("/")
async def stt_websocket(
    websocket: WebSocket,
    authorization: str = Security(get_api_key_ws)
):
    
    await websocket.accept()
    audio_bytes = b""
    language = "en-US"
    sample_rate = 8000
    moonshine = get_stt_model(STT_REPO)

    try:
        last_transmit_idx = 0
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # Accumulate audio bytes
                    audio_bytes += message["bytes"]
                    # Check if enough audio has been received to trigger a final transcription
                    # STT_PAUSE is in ms, sample_rate is in Hz, 2 bytes per int16 sample
                    bytes_per_ms = int(sample_rate * 2 / 1000)
                    while len(audio_bytes) - last_transmit_idx >= STT_PAUSE * bytes_per_ms:
                        chunk = audio_bytes[last_transmit_idx:last_transmit_idx + STT_PAUSE * bytes_per_ms]
                        if chunk:
                            audio_np = np.frombuffer(chunk, dtype=np.int16)
                            transcription = moonshine.stt((sample_rate, audio_np))
                            await websocket.send_json({
                                "type": "transcription",
                                "is_final": True,
                                "alternatives": [{"transcript": transcription, "confidence": 1.0}],
                                "language": language,
                                "channel": 1
                            })
                            last_transmit_idx += len(chunk)
                elif "text" in message:
                    # Handle control messages (e.g., start/stop)
                    control = json.loads(message["text"])
                    print(f"Control message received: {control}")
                    if control.get("type") == "start":
                        language = control.get("language", "en-US")
                        sample_rate = control.get("sampleRateHz", 8000)
                        # You can handle other options here
                    elif control.get("type") == "stop":
                        # On stop, send any remaining audio as a final transcript
                        if len(audio_bytes) > last_transmit_idx:
                            chunk = audio_bytes[last_transmit_idx:]
                            audio_np = np.frombuffer(chunk, dtype=np.int16)
                            transcription = moonshine.stt((sample_rate, audio_np))
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
