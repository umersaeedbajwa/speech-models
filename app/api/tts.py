import os
from fastapi import APIRouter, Response, status
from kokoro import KPipeline
from app.auth import get_api_key
import soundfile as sf
import io
import numpy as np
from dotenv import load_dotenv
from fastapi import Security
from app.models.schemas import TTSRequest
from functools import cache
load_dotenv()

TTS_REPO = os.getenv("TTS_REPO")
lang_map = {
    "en-US": "a", "en-GB": "b", "es": "e", "fr-fr": "f", "hi": "h", "it": "i", "ja": "j", "pt-br": "p", "zh": "z"
}

@cache
def get_pipeline(language):
    lang_code = lang_map.get(language, "a")
    return KPipeline(lang_code=lang_code, repo_id=TTS_REPO)

get_pipeline("en-US")
router = APIRouter()


@router.post("/", status_code=status.HTTP_200_OK)
async def synthesize(
    tts_request: TTSRequest,
    authorization: str = Security(get_api_key)
):
    """
    TTS endpoint for jambonz integration.
    
    Input (JSON body):
        {
            "language": "en-US",      # ISO language code
            "voice": "string",        # Name of voice to use
            "type": "text"|"ssml",   # Input type
            "text": "string"           # Text or SSML to synthesize
        }
    
    Output:
        - 200 OK with audio data in body
        - Content-Type: audio/mpeg | audio/wav | audio/l16;rate=8000 | ...
        - On error: appropriate HTTP error code
    
    Integration:
        - jambonz will POST to this endpoint with the above JSON and an Authorization header
        - Your implementation should authenticate, synthesize speech, and return audio in the requested format
    """
    
    language = tts_request.language
    voice = tts_request.voice
    text = tts_request.text
    pipeline = get_pipeline(language)
    generator = pipeline(text, voice=voice, speed=1, split_pattern=r'\\n+')
    # Concatenate all audio segments
    audio_segments = []
    for i, (gs, ps, audio) in enumerate(generator):
        audio_segments.append(audio)
    if not audio_segments:
        return Response(content=b"", status_code=400, media_type="text/plain")
    audio_concat = np.concatenate(audio_segments)
    buf = io.BytesIO()
    sf.write(buf, audio_concat, 24000, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
