# Voice Models

## Requirements

- Python 3.12
- [Uvicorn](https://www.uvicorn.org/) (ASGI server)
- [pytest](https://docs.pytest.org/en/stable/)

## Setup

1. **Install dependencies**  
   *(Use a virtual environment if possible)*

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   pip install -r requirements.txt
   ```

2. **Set PYTHONPATH**  
   *(For Windows PowerShell)*

   ```powershell
   $env:PYTHONPATH = "."
   ```

## Running the API

Start the FastAPI app with Uvicorn:

```powershell
uvicorn app.main:app --reload
```

## Running Tests

- Run all tests (ignore warnings):

  ```powershell
  pytest -s -W ignore
  ```

- Run a specific test by keyword:

  ```powershell
  pytest -k test_stt_websocket_from_latest_wav -s
  ```

- Run a specific test file:

  ```powershell
  pytest tests/test_2_stt.py -s
  ```

## Running TTS and STT Services Separately or Combined

You can run the TTS and STT APIs either together (combined) or as separate services.

### Combined Service (default)
This exposes both `/tts` and `/stt` endpoints from a single FastAPI app:

```powershell
uvicorn app.main:app --reload
```

- TTS endpoint:   `http://localhost:8000/tts/...`
- STT endpoint:   `http://localhost:8000/stt/...`

### TTS Service Only
Runs only the TTS API on a separate port (e.g., 8001):

```powershell
uvicorn app.tts_service:app --reload --port 8001
```
- TTS endpoint:   `http://localhost:8001/tts/...`

### STT Service Only
Runs only the STT API on a separate port (e.g., 8002):

```powershell
uvicorn app.stt_service:app --reload --port 8002
```
- STT endpoint:   `http://localhost:8002/stt/...`

---

**Tip:**  
Update `requirements.txt` as needed to keep your dependencies up to date.