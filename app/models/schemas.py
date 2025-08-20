from pydantic import BaseModel

class TTSRequest(BaseModel):
    language: str = "en-US"
    voice: str = "af_heart"
    type: str = "text"
    text: str

class TTSResponse(BaseModel):
    audio: bytes

class STTRequest(BaseModel):
    audio: bytes

class STTResponse(BaseModel):
    text: str
