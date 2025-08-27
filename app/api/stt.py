from fastapi import APIRouter, Security, WebSocket
from app.auth import get_api_key_ws
import numpy as np
from dotenv import load_dotenv
import json

load_dotenv()

router = APIRouter()


@router.websocket("/")
async def stt_websocket(
    websocket: WebSocket,
    authorization: str = Security(get_api_key_ws)
):
    await websocket.accept()
    
    language = "en-US"

    # Return transcription result on every Pause and on stop
    # await websocket.send_json({
    #     "type": "transcription",
    #     "is_final": True,
    #     "alternatives": [{"transcript": transcription.strip(), "confidence": 1.0}],
    #     "language": language,
    #     "channel": 1
    # })
    
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    audio_chunk = np.frombuffer(message["bytes"], dtype=np.int16)
                        
                elif "text" in message:
                    control = json.loads(message["text"])
                    
                    if control.get("type") == "start":
                        language = control.get("language", "en-US")
                        sample_rate = control.get("sampleRateHz", 8000)
                    elif control.get("type") == "stop":
                        
                        await websocket.close()
                        break
    except Exception as e:
        await websocket.send_json({"type": "error", "error": str(e)})
        await websocket.close()
