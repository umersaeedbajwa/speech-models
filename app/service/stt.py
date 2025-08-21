
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol

import click
import librosa
import numpy as np
from numpy.typing import NDArray

from app.utils import AudioChunk, audio_to_float32

curr_dir = Path(__file__).parent


class STTModel(Protocol):
    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str: ...


class MoonshineSTT(STTModel):
    def __init__(
        self, model: Literal["moonshine/base", "moonshine/tiny"] = "moonshine/base"
    ):
        try:
            from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Install fastrtc[stt] for speech-to-text and stopword detection support."
            )

        self.model = MoonshineOnnxModel(model_name=model)
        self.tokenizer = load_tokenizer()

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sr, audio_np = audio  # type: ignore
        audio_np = audio_to_float32(audio_np)
        if sr != 16000:
            audio_np: NDArray[np.float32] = librosa.resample(
                audio_np, orig_sr=sr, target_sr=16000
            )
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        tokens = self.model.generate(audio_np)
        return self.tokenizer.decode_batch(tokens)[0]


@lru_cache
def get_stt_model(
    model: Literal["moonshine/base", "moonshine/tiny"] = "moonshine/base",
) -> STTModel:
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    m = MoonshineSTT(model)
    from moonshine_onnx import load_audio
    test_file = curr_dir / ".." / ".." / "tests" / "test_file.wav"
    test_file = test_file.resolve()
    if test_file.exists():
        audio = load_audio(str(test_file))
        print(click.style("INFO", fg="green") + ":\t  Warming up STT model.")

        m.stt((24000, audio))
        print(click.style("INFO", fg="green") + ":\t  STT model warmed up.")
    else:
        print(click.style("WARNING", fg="yellow") + f":\t  No test_file.wav found, skipping STT model warmup.")
    return m


def stt_for_chunks(
    stt_model: STTModel,
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chunks: list[AudioChunk],
) -> str:
    sr, audio_np = audio
    return " ".join(
        [
            stt_model.stt((sr, audio_np[chunk["start"] : chunk["end"]]))
            for chunk in chunks
        ]
    )