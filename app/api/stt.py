import os
from typing import Literal
from fastapi import APIRouter, Security, WebSocket
from app.auth import get_api_key_ws
import numpy as np
from dotenv import load_dotenv
import json

load_dotenv()

STT_REPO = os.getenv("STT_REPO")

router = APIRouter()


@router.websocket("/")
async def stt_websocket(
    websocket: WebSocket,
    authorization: str = Security(get_api_key_ws)
):
    await websocket.accept()
    
    language = "en-US"
    
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # Convert bytes to numpy array
                    audio_chunk = np.frombuffer(message["bytes"], dtype=np.int16)
                    
                    # Process with pause detection
                    
                    # Send transcript when pause is detected
                    if pause_detected and captions:
                        await websocket.send_json({
                            "type": "transcription",
                            "is_final": True,
                            "alternatives": [{"transcript": captions, "confidence": 1.0}],
                            "language": language,
                            "channel": 1
                        })
                        
                elif "text" in message:
                    control = json.loads(message["text"])
                    print(f"Control message received: {control}")
                    if control.get("type") == "start":
                        language = control.get("language", "en-US")
                        # Reset state properly
                        state.sample_rate = control.get("sampleRateHz", 8000)
                    elif control.get("type") == "stop":
                        # Send any remaining captions
                        if state.started_talking and len(state.buffer) > 0:
                            try:
                                moonshine = get_stt_model()
                                transcription = moonshine.stt((state.sample_rate, state.buffer))
                                if transcription and transcription.strip():
                                    await websocket.send_json({
                                        "type": "transcription",
                                        "is_final": True,
                                        "alternatives": [{"transcript": transcription.strip(), "confidence": 1.0}],
                                        "language": language,
                                        "channel": 1
                                    })
                            except Exception as e:
                                print(f"Final STT error: {e}")
                        await websocket.close()
                        break
    except Exception as e:
        await websocket.send_json({"type": "error", "error": str(e)})
        await websocket.close()
