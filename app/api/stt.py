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
    """WebSocket endpoint for STT with pause detection using ReplyOnPause"""
    await websocket.accept()
    
    language = "en-US"
    sample_rate = 8000
    audio_buffer = []
    use_streaming = False
    reply_handler = None
    
    def transcribe_on_pause(audio: tuple[int, np.ndarray]):
        """Generator function that performs STT when pause is detected"""
        try:
            sample_rate, audio_data = audio
            
            # Initialize STT model
            stt_model = get_stt_model("moonshine/base")
            
            # Perform speech-to-text
            transcription = stt_model.stt(audio)
            
            # Create transcription result
            result = {
                "type": "transcription",
                "is_final": True,
                "alternatives": [{"transcript": transcription.strip(), "confidence": 1.0}],
                "language": language,
                "channel": 1,
                "triggered_by": "pause"
            }
            
            # Yield the result using AdditionalOutputs (non-audio data)
            yield AdditionalOutputs(result)
            
        except Exception as e:
            error_result = {"type": "error", "error": f"STT Error: {str(e)}"}
            yield AdditionalOutputs(error_result)
    
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    audio_data = message["bytes"]
                    
                    if use_streaming and reply_handler:
                        # Process audio data through ReplyOnPause for streaming
                        try:
                            audio_chunk = np.frombuffer(audio_data, dtype=np.int16)
                            # Convert to float32 for ReplyOnPause
                            audio_float = audio_chunk.astype(np.float32) / 32768.0
                            
                            # Send audio frame to ReplyOnPause handler
                            reply_handler.receive((sample_rate, audio_float))
                            
                            # Check if there's output from the handler
                            output = reply_handler.emit()
                            if output is not None:
                                # Output contains AdditionalOutputs with transcription result
                                if hasattr(output, 'value'):
                                    result = output.value
                                    await websocket.send_json(result)
                                else:
                                    await websocket.send_json(output)
                            
                        except Exception as e:
                            await websocket.send_json({"type": "error", "error": f"ReplyOnPause Error: {str(e)}"})
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
                            try:
                                # Initialize ReplyOnPause handler for streaming
                                reply_handler = ReplyOnPause(
                                    fn=transcribe_on_pause,
                                    input_sample_rate=sample_rate,
                                    output_sample_rate=sample_rate,
                                    expected_layout="mono",
                                    can_interrupt=True
                                )
                                
                                # Start up the handler
                                reply_handler.start_up()
                            except Exception as e:
                                await websocket.send_json({"type": "error", "error": f"ReplyOnPause init error: {str(e)}"})
                                use_streaming = False
                        else:
                            use_streaming = False
                            audio_buffer = []
                        
                    elif control.get("type") == "stop":
                        if use_streaming and reply_handler:
                            try:
                                # Trigger final response if any audio is buffered
                                reply_handler.trigger_response()
                                
                                # Get final output
                                output = reply_handler.emit()
                                if output is not None:
                                    # Send final transcription if available
                                    if hasattr(output, 'value'):
                                        result = output.value
                                        await websocket.send_json(result)
                                    else:
                                        await websocket.send_json(output)
                            except Exception as e:
                                await websocket.send_json({"type": "error", "error": f"ReplyOnPause stop error: {str(e)}"})
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
