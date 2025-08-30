from fastapi import FastAPI
from app.api import tts

app = FastAPI()
app.include_router(tts.router, tags=["TTS"])