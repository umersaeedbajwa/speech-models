from fastapi import APIRouter, Security, WebSocket, Response, status
from app.auth import get_api_key_ws, get_api_key
from app.models.schemas import STTRequest, STTResponse
import numpy as np
from dotenv import load_dotenv
import json
import soundfile as sf
import io
import time
from fastrtc import (
    ReplyOnPause,
    ReplyOnStopWords,
    Stream,
    get_stt_model,
    get_twilio_turn_credentials,
    AdditionalOutputs,
    WebRTCError,
    get_current_context,
    get_hf_turn_credentials
)

load_dotenv()

router = APIRouter()


@router.websocket("/")
async def stt_websocket_with_pause(
    websocket: WebSocket,
    authorization: str = Security(get_api_key_ws)
):
    """WebSocket endpoint for STT with pause detection"""
    await websocket.accept()
    
    language = "en-US"
    sample_rate = 8000
    audio_buffer = []
    use_streaming = False
    streaming_buffer = []
    silence_threshold = 0.01  # RMS threshold for silence detection
    silence_duration = 0.0
    max_silence_duration = 1.0  # 1 second of silence triggers transcription
    last_audio_time = time.time()
    
    async def process_streaming_audio():
        """Process accumulated streaming audio and perform STT"""
        if len(streaming_buffer) == 0:
            return
            
        try:
            # Concatenate all audio chunks
            complete_audio = np.concatenate(streaming_buffer)
            
            # Initialize STT model
            stt_model = get_stt_model("moonshine/base")
            
            # Perform speech-to-text
            transcription = stt_model.stt((sample_rate, complete_audio))
            
            # Send transcription result
            await websocket.send_json({
                "type": "transcription",
                "is_final": True,
                "alternatives": [{"transcript": transcription.strip(), "confidence": 1.0}],
                "language": language,
                "channel": 1,
                "triggered_by": "pause"
            })
            
            # Clear the streaming buffer
            streaming_buffer.clear()
            
        except Exception as e:
            await websocket.send_json({"type": "error", "error": f"Streaming STT Error: {str(e)}"})
    
    def is_silence(audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk is mostly silence"""
        if len(audio_chunk) == 0:
            return True
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms < silence_threshold
    
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    audio_data = message["bytes"]
                    
                    if use_streaming:
                        # Process audio data for streaming with pause detection
                        try:
                            audio_chunk = np.frombuffer(audio_data, dtype=np.int16)
                            # Convert to float32 
                            audio_float = audio_chunk.astype(np.float32) / 32768.0
                            
                            current_time = time.time()
                            
                            # Check if this chunk is silence
                            if is_silence(audio_float):
                                silence_duration += current_time - last_audio_time
                                
                                # If we've had enough silence and have audio to process
                                if silence_duration >= max_silence_duration and len(streaming_buffer) > 0:
                                    await process_streaming_audio()
                                    silence_duration = 0.0
                            else:
                                # Reset silence duration and add audio to buffer
                                silence_duration = 0.0
                                streaming_buffer.append(audio_float)
                            
                            last_audio_time = current_time
                            
                        except Exception as e:
                            await websocket.send_json({"type": "error", "error": f"Streaming Error: {str(e)}"})
                    else:
                        # Buffer audio data for complete file processing
                        audio_buffer.append(audio_data)
                        
                elif "text" in message:
                    control = json.loads(message["text"])
                    
                    if control.get("type") == "start":
                        language = control.get("language", "en-US")
                        sample_rate = control.get("sampleRateHz", 8000)
                        
                        # Check if we should use streaming mode
                        streaming_mode = control.get("streaming", False)
                        
                        if streaming_mode:
                            use_streaming = True
                            streaming_buffer = []
                            silence_duration = 0.0
                            last_audio_time = time.time()
                        else:
                            use_streaming = False
                            audio_buffer = []
                        
                    elif control.get("type") == "stop":
                        if use_streaming:
                            # Process any remaining audio in the streaming buffer
                            if len(streaming_buffer) > 0:
                                await process_streaming_audio()
                        elif not use_streaming and audio_buffer:
                            # Process complete audio file
                            try:
                                complete_audio = b''.join(audio_buffer)
                                
                                # Try to read as WAV file first
                                try:
                                    with sf.SoundFile(io.BytesIO(complete_audio)) as f:
                                        audio_array = f.read(dtype=np.float32)  # Read as float32
                                        actual_sample_rate = f.samplerate
                                        if len(audio_array.shape) > 1:
                                            audio_array = audio_array.mean(axis=1)  # Convert stereo to mono
                                except Exception as sf_error:
                                    # If not WAV, treat as raw PCM data
                                    audio_array = np.frombuffer(complete_audio, dtype=np.int16).astype(np.float32) / 32768.0
                                    actual_sample_rate = sample_rate
                                
                                # Initialize STT model
                                stt_model = get_stt_model("moonshine/base")
                                
                                # Perform speech-to-text (moonshine expects tuple of (sample_rate, audio_array))
                                transcription = stt_model.stt((actual_sample_rate, audio_array))
                                
                                # Send transcription result
                                await websocket.send_json({
                                    "type": "transcription",
                                    "is_final": True,
                                    "alternatives": [{"transcript": transcription.strip(), "confidence": 1.0}],
                                    "language": language,
                                    "channel": 1,
                                    "triggered_by": "complete_audio"
                                })
                                
                            except Exception as e:
                                await websocket.send_json({"type": "error", "error": f"Complete audio STT error: {str(e)}"})
                        
                        await websocket.close()
                        break
                        
            elif message["type"] == "websocket.disconnect":
                break
                
    except WebRTCError as e:
        await websocket.send_json({"type": "error", "error": f"WebRTC Error: {str(e)}"})
        await websocket.close()
    except Exception as e:
        await websocket.send_json({"type": "error", "error": str(e)})
        await websocket.close()
