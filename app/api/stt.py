from fastapi import APIRouter, Security, WebSocket
from app.auth import get_api_key_ws
import numpy as np
import json
import logging
from fastrtc import get_stt_model
from starlette.websockets import WebSocketState
import webrtcvad

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize STT model
stt_model = get_stt_model("moonshine/base")
logger.info("STT model initialized")

router = APIRouter()


@router.websocket("/")
async def jambonz_stt_websocket(
    websocket: WebSocket,
    authorization: str = Security(get_api_key_ws)
):
    """Jambonz-compatible WebSocket STT endpoint using moonshine"""
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    # Connection state
    language = "en-US"
    sample_rate = 8000
    interim_results = False
    audio_buffer = []
    vad = webrtcvad.Vad(3)  # High sensitivity
    silence_duration = 0.0
    frame_duration = 0.02  # 20ms per frame
    min_silence = 1.5  # seconds

    while True:
        try:
            message = await websocket.receive()
        except RuntimeError as e:
            logger.error(f"WebSocket disconnect or receive error: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Server error: {str(e)}"
                })
            except Exception:
                pass
            await websocket.close()
            break

        if "bytes" in message:
            frame = message["bytes"]
            audio_buffer.append(frame)

            # VAD expects 16-bit mono PCM, 8kHz, frame length 10/20/30ms
            # Ensure frame is correct size for VAD (e.g., 20ms = 160 samples at 8kHz = 320 bytes)
            try:
                is_speech = vad.is_speech(frame, sample_rate)
            except Exception as e:
                logger.warning(f"VAD error: {e}, treating as speech")
                is_speech = True

            if not is_speech:
                silence_duration += frame_duration
            else:
                silence_duration = 0.0

            logger.info(f"Is speech: {is_speech}")

            # If silence longer than threshold, finalize transcription
            if silence_duration >= min_silence and audio_buffer:
                try:
                    complete_audio = b''.join(audio_buffer)
                    logger.info(f"Processing {len(complete_audio)} bytes of buffered audio (silence-triggered)")
                    audio_array = np.frombuffer(complete_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    transcription = stt_model.stt((sample_rate, audio_array))
                    logger.info(f"Transcription (silence-triggered): {transcription}")

                    response = {
                        "type": "transcription",
                        "is_final": True,
                        "alternatives": [{
                            "transcript": transcription.strip(),
                            "confidence": 1.0
                        }],
                        "channel": 1,
                        "language": language
                    }
                    await websocket.send_json(response)
                except Exception as e:
                    logger.error(f"Error during STT on silence: {e}")
                    try:
                        await websocket.send_json({"type": "error", "error": str(e)})
                    except Exception:
                        pass

                # Reset buffer and silence timer for next utterance
                audio_buffer = []
                silence_duration = 0.0

        elif "text" in message:
            control = json.loads(message["text"])
            logger.info(f"Received control message: {control}")

            if control.get("type") == "start":
                language = control.get("language", "en-US")
                sample_rate = control.get("sampleRateHz", 8000)
                interim_results = control.get("interimResults", False)
                logger.info(f"Starting STT - Language: {language}, Sample Rate: {sample_rate}")
                audio_buffer = []
                silence_duration = 0.0

            elif control.get("type") == "stop":
                logger.info("Stop message received")
                if audio_buffer:
                    try:
                        complete_audio = b''.join(audio_buffer)
                        logger.info(f"Processing {len(complete_audio)} bytes of buffered audio (stop-triggered)")
                        audio_array = np.frombuffer(complete_audio, dtype=np.int16).astype(np.float32) / 32768.0
                        transcription = stt_model.stt((sample_rate, audio_array))
                        logger.info(f"Transcription result: {transcription}")

                        response = {
                            "type": "transcription",
                            "is_final": True,
                            "alternatives": [{
                                "transcript": transcription.strip(),
                                "confidence": 1.0
                            }],
                            "channel": 1,
                            "language": language
                        }
                        await websocket.send_json(response)
                    except Exception as e:
                        logger.error(f"Error processing buffered audio: {e}")
                        try:
                            await websocket.send_json({
                                "type": "error",
                                "error": f"Audio processing error: {str(e)}"
                            })
                        except Exception:
                            pass
                # Do not close the WebSocket, just reset buffer and silence timer
                audio_buffer = []
                silence_duration = 0.0
    logger.info("WebSocket handler cleanup completed")
