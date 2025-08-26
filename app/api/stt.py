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
STT_CHUNK_DURATION = float(os.getenv("STT_CHUNK_DURATION", 0.6))  # 600ms like ReplyOnPause
STT_STARTED_THRESHOLD = float(os.getenv("STT_STARTED_THRESHOLD", 0.2))  # 200ms speech to start
STT_SPEECH_THRESHOLD = float(os.getenv("STT_SPEECH_THRESHOLD", 0.1))  # 100ms speech to continue

router = APIRouter()


class AudioState:
    """Simple state management similar to ReplyOnPause AppState"""
    def __init__(self):
        self.buffer = np.array([], dtype=np.int16)
        self.captions = ""
        self.started_talking = False
        self.last_speech_time = 0
        self.sample_rate = 8000


def has_speech_content(audio_chunk: np.ndarray, sample_rate: int) -> bool:
    """Check if audio chunk contains speech using STT (more accurate than energy)"""
    if len(audio_chunk) == 0:
        return False
    
    # Only check speech for reasonably sized chunks to avoid STT errors
    min_samples = int(sample_rate * 0.1)  # 100ms minimum
    if len(audio_chunk) < min_samples:
        return False
    
    try:
        moonshine = get_stt_model(STT_REPO)
        transcription = moonshine.stt((sample_rate, audio_chunk))
        return bool(transcription and transcription.strip())
    except:
        # If STT fails, assume no speech
        return False


def determine_pause(audio_chunk: np.ndarray, state: AudioState) -> bool:
    """
    Pause detection logic matching ReplyOnPause.determine_pause
    Returns True if pause detected, False otherwise
    """
    duration = len(audio_chunk) / state.sample_rate
    
    # Only process chunks that meet minimum duration (like ReplyOnPause)
    if duration >= STT_CHUNK_DURATION:
        # Use STT to get speech duration (equivalent to dur_vad)
        has_speech = has_speech_content(audio_chunk, state.sample_rate)
        dur_stt = duration if has_speech else 0.0  # Binary: full duration or none
        
        print(f"Chunk duration: {duration:.2f}s, STT speech duration: {dur_stt:.2f}s")
        
        # Check if user started talking (like started_talking_threshold check)
        if dur_stt > STT_STARTED_THRESHOLD and not state.started_talking:
            state.started_talking = True
            print("Started talking")
        
        # If user started talking, accumulate speech in buffer (like state.stream)
        if state.started_talking:
            if len(state.buffer) == 0:
                state.buffer = audio_chunk
            else:
                state.buffer = np.concatenate([state.buffer, audio_chunk])
            
            # Check if continuous speech limit has been reached (optional feature)
            # current_duration = len(state.buffer) / state.sample_rate
            # if current_duration >= max_continuous_speech_s:
            #     return True
        
        # Check if a pause has been detected (like speech_threshold check)
        if dur_stt < STT_SPEECH_THRESHOLD and state.started_talking:
            print(f"Pause detected: STT speech duration {dur_stt:.2f}s < threshold {STT_SPEECH_THRESHOLD}")
            return True
    
    return False


def stream_stt_with_pause_detection(audio_chunk: np.ndarray, state: AudioState) -> tuple[str, bool]:
    """
    Process audio chunk and detect pauses using ReplyOnPause-style logic
    Returns: (updated_captions, pause_detected)
    """
    # Use ReplyOnPause-style pause detection
    pause_detected = determine_pause(audio_chunk, state)
    
    if pause_detected:
        print(f"Processing accumulated buffer of {len(state.buffer)} samples")
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
