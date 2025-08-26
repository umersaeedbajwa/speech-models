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
STT_SILENCE_THRESHOLD = int(os.getenv("STT_SILENCE_THRESHOLD", 100))  # Ensure integer

router = APIRouter()


class AudioState:
    """Simple state management similar to ReplyOnPause AppState"""
    def __init__(self):
        self.buffer = np.array([], dtype=np.int16)
        self.captions = ""
        self.started_talking = False
        self.last_speech_time = 0
        self.sample_rate = 8000


def has_speech(audio_chunk: np.ndarray, threshold: int = STT_SILENCE_THRESHOLD) -> bool:
    """Simple voice activity detection using energy threshold"""
    if len(audio_chunk) == 0:
        return False
    energy = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
    return energy > threshold


def stream_stt_with_pause_detection(audio_chunk: np.ndarray, state: AudioState) -> tuple[str, bool]:
    """
    Process audio chunk and detect pauses - inspired by ReplyOnPause.determine_pause
    Returns: (updated_captions, pause_detected)
    """
    current_time = asyncio.get_event_loop().time()
    
    # Add chunk to buffer
    if len(state.buffer) == 0:
        state.buffer = audio_chunk
    else:
        state.buffer = np.concatenate([state.buffer, audio_chunk])
    
    # Check for speech activity
    if has_speech(audio_chunk):
        if not state.started_talking:
            state.started_talking = True
            print("Started talking")
        state.last_speech_time = current_time
        return state.captions, False  # No pause while talking
    
    # Check for pause after speech started
    if state.started_talking:
        silence_duration = (current_time - state.last_speech_time) * 1000
        if silence_duration >= STT_PAUSE:
            print(f"Detected pause after {silence_duration}ms")
            # Process accumulated buffer with STT
            try:
                moonshine = get_stt_model(STT_REPO)
                transcription = moonshine.stt((state.sample_rate, state.buffer))
                if transcription and transcription.strip():
                    new_captions = transcription.strip()
                    
                    # Reset state for next speech segment
                    state.buffer = np.array([], dtype=np.int16)
                    state.started_talking = False
                    
                    return new_captions, True
                else:
                    # No transcription but reset state anyway
                    state.buffer = np.array([], dtype=np.int16)
                    state.started_talking = False
            except Exception as e:
                print(f"STT error: {e}")
                # Reset state even on error
                state.buffer = np.array([], dtype=np.int16)
                state.started_talking = False
    
    return state.captions, False


@router.websocket("/")
async def stt_websocket(
    websocket: WebSocket,
    authorization: str = Security(get_api_key_ws)
):
    await websocket.accept()
    
    state = AudioState()
    language = "en-US"
    
    try:
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # Convert bytes to numpy array
                    audio_chunk = np.frombuffer(message["bytes"], dtype=np.int16)
                    
                    # Process with pause detection
                    captions, pause_detected = stream_stt_with_pause_detection(audio_chunk, state)
                    
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
                    if control.get("type") == "start":
                        language = control.get("language", "en-US")
                        # Reset state properly
                        state = AudioState()
                        state.sample_rate = control.get("sampleRateHz", 8000)
                    elif control.get("type") == "stop":
                        # Send any remaining captions
                        if state.started_talking and len(state.buffer) > 0:
                            try:
                                moonshine = get_stt_model(STT_REPO)
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
