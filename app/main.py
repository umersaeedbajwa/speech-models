from fastapi import FastAPI
from app.api import tts, stt

app = FastAPI()

@app.get("/")
def root():
	return {"message": "Welcome to the Voice Models API"}
app.include_router(tts.router, prefix="/", tags=["TTS"])
app.include_router(stt.router, prefix="/", tags=["STT"])
