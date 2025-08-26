import os
from typing import Literal
from fastapi import APIRouter, Security, WebSocket
from app.auth import get_api_key_ws
import numpy as np
from dotenv import load_dotenv
import json
import asyncio
from app.service.stt import get_stt_model

load_dotenv()


STT_REPO = os.getenv("STT_REPO")
STT_PAUSE = int(os.getenv("STT_PAUSE", 1500))  # Ensure integer
STT_SILENCE_THRESHOLD = int(os.getenv("STT_SILENCE_THRESHOLD", 200))  # Ensure integer

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
    last_transcript_time = 0
    last_audio_activity_time = asyncio.get_event_loop().time()
    import time

    def has_audio_content(audio_data):
        """Check if audio data contains actual speech (not just silence)"""
        if len(audio_data) == 0:
            return False
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        energy = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))
        return energy > STT_SILENCE_THRESHOLD  # Threshold for detecting speech vs silence

    try:
        while True:
            message = await websocket.receive()
            current_time = asyncio.get_event_loop().time()
            
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # Accumulate audio bytes
                    audio_bytes += message["bytes"]
                    
                    # Check if this chunk has audio content
                    if has_audio_content(message["bytes"]):
                        last_audio_activity_time = current_time
                    
                    # Check if we've had silence for STT_PAUSE duration
                    silence_duration = (current_time - last_audio_activity_time) * 1000  # Convert to ms
                    if silence_duration >= STT_PAUSE and len(audio_bytes) > last_transcript_time:
                        new_audio = audio_bytes[last_transcript_time:]
                        if has_audio_content(new_audio):
                            audio_np = np.frombuffer(new_audio, dtype=np.int16)
                            transcription = moonshine.stt((sample_rate, audio_np))
                            print(f"Silence timeout transcription: {transcription}")
                            
                            await websocket.send_json({
                                "type": "transcription",
                                "is_final": True,
                                "alternatives": [{"transcript": transcription, "confidence": 1.0}],
                                "language": language,
                                "channel": 1
                            })
                            last_transcript_time = len(audio_bytes)
                            
                elif "text" in message:
                    # Handle control messages (e.g., start/stop)
                    control = json.loads(message["text"])
                    print(f"Control message received: {control}")
                    if control.get("type") == "start":
                        language = control.get("language", "en-US")
                        sample_rate = control.get("sampleRateHz", 8000)
                        # Reset audio buffer on start
                        audio_bytes = b""
                        last_transcript_time = 0
                        last_audio_activity_time = current_time
                    elif control.get("type") == "stop":
                        # Send final transcript for any remaining audio
                        if audio_bytes and has_audio_content(audio_bytes[last_transcript_time:]):
                            new_audio = audio_bytes[last_transcript_time:]
                            audio_np = np.frombuffer(new_audio, dtype=np.int16)
                            transcription = moonshine.stt((sample_rate, audio_np))
                            print(f"Final transcription: {transcription}")
                            
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
