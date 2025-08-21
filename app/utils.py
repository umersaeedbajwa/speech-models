
import io
import json
import logging
import tempfile
import warnings
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

import numpy as np
from fastapi import WebSocket
from numpy.typing import NDArray
# from pydub import AudioSegment

logger = logging.getLogger(__name__)


AUDIO_PTIME = 0.020


class AudioChunk(TypedDict):
    start: int
    end: int

class AdditionalOutputs:
    def __init__(self, *args) -> None:
        self.args = args


class CloseStream:
    def __init__(self, msg: str = "Stream closed") -> None:
        self.msg = msg


class DataChannel(Protocol):
    def send(self, message: str) -> None: ...


def create_message(
    type: Literal[
        "send_input",
        "end_stream",
        "fetch_output",
        "stopword",
        "error",
        "warning",
        "log",
        "update_connection",
    ],
    data: list[Any] | str,
) -> str:
    return json.dumps({"type": type, "data": data})


current_channel: ContextVar[DataChannel | None] = ContextVar(
    "current_channel", default=None
)


@dataclass
class Context:
    webrtc_id: str
    websocket: WebSocket | None = None


current_context: ContextVar[Context | None] = ContextVar(
    "current_context", default=None
)


# def audio_to_bytes(audio: tuple[int, NDArray[np.int16 | np.float32]]) -> bytes:
#     """
#     Convert an audio tuple containing sample rate and numpy array data into bytes.

#     Parameters
#     ----------
#     audio : tuple[int, np.ndarray]
#         A tuple containing:
#             - sample_rate (int): The audio sample rate in Hz
#             - data (np.ndarray): The audio data as a numpy array

#     Returns
#     -------
#     bytes
#         The audio data encoded as bytes, suitable for transmission or storage

#     Example
#     -------
#     >>> sample_rate = 44100
#     >>> audio_data = np.array([0.1, -0.2, 0.3])  # Example audio samples
#     >>> audio_tuple = (sample_rate, audio_data)
#     >>> audio_bytes = audio_to_bytes(audio_tuple)
#     """
#     audio_buffer = io.BytesIO()
#     segment = AudioSegment(
#         audio[1].tobytes(),
#         frame_rate=audio[0],
#         sample_width=audio[1].dtype.itemsize,
#         channels=1,
#     )
#     segment.export(audio_buffer, format="mp3")
#     return audio_buffer.getvalue()


# def audio_to_file(audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
#     """
#     Save an audio tuple containing sample rate and numpy array data to a file.

#     Parameters
#     ----------
#     audio : tuple[int, np.ndarray]
#         A tuple containing:
#             - sample_rate (int): The audio sample rate in Hz
#             - data (np.ndarray): The audio data as a numpy array

#     Returns
#     -------
#     str
#         The path to the saved audio file

#     Example
#     -------
#     >>> sample_rate = 44100
#     >>> audio_data = np.array([0.1, -0.2, 0.3])  # Example audio samples
#     >>> audio_tuple = (sample_rate, audio_data)
#     >>> file_path = audio_to_file(audio_tuple)
#     >>> print(f"Audio saved to: {file_path}")
#     """
#     bytes_ = audio_to_bytes(audio)
#     with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
#         f.write(bytes_)
#     return f.name


def audio_to_float32(
    audio: NDArray[np.int16 | np.float32] | tuple[int, NDArray[np.int16 | np.float32]],
) -> NDArray[np.float32]:
    """
    Convert an audio tuple containing sample rate (int16) and numpy array data to float32.

    Parameters
    ----------
    audio : np.ndarray
        The audio data as a numpy array

    Returns
    -------
    np.ndarray
        The audio data as a numpy array with dtype float32

    Example
    -------
    >>> audio_data = np.array([0.1, -0.2, 0.3])  # Example audio samples
    >>> audio_float32 = audio_to_float32(audio_data)
    """
    if isinstance(audio, tuple):
        warnings.warn(
            UserWarning(
                "Passing a (sr, audio) tuple to audio_to_float32() is deprecated "
                "and will be removed in a future release. Pass only the audio array."
            ),
            stacklevel=2,  # So that the warning points to the user's code
        )
        _sr, audio = audio

    if audio.dtype == np.int16:
        # Divide by 32768.0 so that the values are in the range [-1.0, 1.0).
        # 1.0 can actually never be reached because the int16 range is [-32768, 32767].
        return audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.float32:
        return audio  # type: ignore
    else:
        raise TypeError(f"Unsupported audio data type: {audio.dtype}")


def audio_to_int16(
    audio: NDArray[np.int16 | np.float32] | tuple[int, NDArray[np.int16 | np.float32]],
) -> NDArray[np.int16]:
    """
    Convert an audio tuple containing sample rate and numpy array data to int16.

    Parameters
    ----------
    audio : np.ndarray
        The audio data as a numpy array

    Returns
    -------
    np.ndarray
        The audio data as a numpy array with dtype int16

    Example
    -------
    >>> audio_data = np.array([0.1, -0.2, 0.3], dtype=np.float32)  # Example audio samples
    >>> audio_int16 = audio_to_int16(audio_data)
    """
    if isinstance(audio, tuple):
        warnings.warn(
            UserWarning(
                "Passing a (sr, audio) tuple to audio_to_float32() is deprecated "
                "and will be removed in a future release. Pass only the audio array."
            ),
            stacklevel=2,  # So that the warning points to the user's code
        )
        _sr, audio = audio

    if audio.dtype == np.int16:
        return audio  # type: ignore
    elif audio.dtype == np.float32:
        # Convert float32 to int16 by scaling to the int16 range.
        # Multiply by 32767 and not 32768 so that int16 doesn't overflow.
        return (audio * 32767.0).astype(np.int16)
    else:
        raise TypeError(f"Unsupported audio data type: {audio.dtype}")

