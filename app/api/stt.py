from fastapi import APIRouter, Security, WebSocket
from app.auth import get_api_key_ws
import numpy as np
import json
import logging
from fastrtc import get_stt_model
from starlette.websockets import WebSocketState

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
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                audio_data = message["bytes"]
                audio_buffer.append(audio_data)
                    
            elif "text" in message:
                control = json.loads(message["text"])
                logger.info(f"Received control message: {control}")
                
                if control.get("type") == "start":
                    language = control.get("language", "en-US")
                    sample_rate = control.get("sampleRateHz", 8000)
                    interim_results = control.get("interimResults", False)
                    logger.info(f"Starting STT - Language: {language}, Sample Rate: {sample_rate}, Interim Results: {interim_results}")
                    audio_buffer = []
                        
                elif control.get("type") == "stop":
                    logger.info("Stop message received")
                    
                    if audio_buffer:
                        try:
                            complete_audio = b''.join(audio_buffer)
                            logger.info(f"Processing {len(complete_audio)} bytes of buffered audio")
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
                            
                            if websocket.client_state == WebSocketState.CONNECTED:
                                try:
                                    await websocket.send_json(response)
                                except Exception as e:
                                    logger.error(f"Error sending transcription: {e}")
                            else:
                                logger.error("WebSocket is not connected. Unable to send transcription.")
                            
                        except Exception as e:
                            logger.error(f"Error processing buffered audio: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "error": f"Audio processing error: {str(e)}"
                            })
                    await websocket.close()
                    logger.info("WebSocket connection closed")
                    break
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await websocket.send_json({
            "type": "error",
            "error": f"Server error: {str(e)}"
        })
        await websocket.close()
    finally:
        logger.info("WebSocket handler cleanup completed")
