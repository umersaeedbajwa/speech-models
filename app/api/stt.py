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
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                        transcription = moonshine.stt((sample_rate, audio_np))
                        print(f"Transcription: {transcription}")
                        
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
