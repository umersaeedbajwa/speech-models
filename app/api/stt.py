from fastapi import APIRouter, Security, WebSocket
from app.auth import get_api_key_ws
import numpy as np
import json
import logging
from fastrtc import get_stt_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    stt_model = None
    
    try:
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    # Handle binary audio data - always buffer for Jambonz compatibility
                    audio_data = message["bytes"]
                    audio_buffer.append(audio_data)
                        
                elif "text" in message:
                    # Handle JSON control messages
                    try:
                        control = json.loads(message["text"])
                        logger.info(f"Received control message: {control}")
                        
                        if control.get("type") == "start":
                            # Extract Jambonz start parameters
                            language = control.get("language", "en-US")
                            sample_rate = control.get("sampleRateHz", 8000)
                            interim_results = control.get("interimResults", False)
                            
                            logger.info(f"Starting STT - Language: {language}, Sample Rate: {sample_rate}")
                            
                            # Initialize STT model
                            stt_model = get_stt_model("moonshine/base")
                            audio_buffer = []
                                
                        elif control.get("type") == "stop":
                            logger.info("Stop message received")
                            
                            if audio_buffer and stt_model:
                                # Process buffered audio for final transcription
                                try:
                                    complete_audio = b''.join(audio_buffer)
                                    logger.info(f"Processing {len(complete_audio)} bytes of buffered audio")
                                    
                                    # Treat as raw LINEAR16 PCM data
                                    audio_array = np.frombuffer(complete_audio, dtype=np.int16).astype(np.float32) / 32768.0
                                    
                                    # Perform STT
                                    transcription = stt_model.stt((sample_rate, audio_array))
                                    
                                    if transcription and transcription.strip():
                                        # Send Jambonz-compatible response
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
                                        
                                        logger.info(f"Sending final transcription: {response}")
                                        await websocket.send_json(response)
                                    
                                except Exception as e:
                                    logger.error(f"Error processing buffered audio: {e}")
                                    await websocket.send_json({
                                        "type": "error",
                                        "error": f"Audio processing error: {str(e)}"
                                    })
                            
                            # Close connection after stop
                            await websocket.close()
                            logger.info("WebSocket connection closed")
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in control message: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Invalid JSON: {str(e)}"
                        })
                        
            elif message["type"] == "websocket.disconnect":
                logger.info("WebSocket disconnected")
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
