from fastapi import FastAPI
from app.api import tts, stt

app = FastAPI()

@app.get("/")
def root():
	return {"message": "Welcome to the Voice Models API"}
app.include_router(tts.router, prefix="/tts", tags=["TTS"])
app.include_router(stt.router, prefix="/stt", tags=["STT"])
