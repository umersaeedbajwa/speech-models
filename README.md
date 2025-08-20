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

---

**Tip:**  
Update `requirements.txt` as needed to keep