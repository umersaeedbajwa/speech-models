from fastapi import FastAPI
from app.api import stt

app = FastAPI()
app.include_router(stt.router, tags=["STT"])