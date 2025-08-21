from fastapi import APIRouter, Response, status
from app.auth import get_api_key
import soundfile as sf
import io
from fastapi import Security
from app.models.schemas import TTSRequest
from app.service.tts import get_tts_model,KokoroTTSOptions

router = APIRouter()

@router.post("/", status_code=status.HTTP_200_OK)
async def synthesize(
    tts_request: TTSRequest,
    authorization: str = Security(get_api_key)
):
    
    language = tts_request.language
    voice = tts_request.voice
    text = tts_request.text

    model = get_tts_model()
    options = KokoroTTSOptions(voice=voice, speed=1.0, lang=language)
    sample_rate, audio = model.tts(text, options)

    if audio is None:
        return Response(content=b"", status_code=400, media_type="text/plain")
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")
