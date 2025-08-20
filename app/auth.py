from fastapi import Security, HTTPException, status, WebSocket
from fastapi.security.api_key import APIKeyHeader
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY", "testkey")
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_api_key(
    api_key: str = Security(api_key_header)
):
    # For HTTP endpoints
    has_bearer = api_key.startswith("Bearer ") if api_key else False
    key_value = api_key.split(" ")[1] if has_bearer else None
    is_valid = key_value == API_KEY if key_value else False

    if not api_key or not has_bearer or not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Invalid or missing API key",
                "api_key": key_value,
                "has_bearer": has_bearer,
                "is_valid": is_valid,
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    return api_key

async def get_api_key_ws(websocket: WebSocket):
    # For WebSocket endpoints
    auth_header = websocket.headers.get("authorization")
    has_bearer = auth_header.lower().startswith("bearer ") if auth_header else False
    key_value = auth_header.split(" ")[1] if has_bearer else None
    is_valid = key_value == API_KEY if key_value else False

    if not auth_header or not has_bearer or not is_valid:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Invalid or missing API key",
                "api_key": key_value,
                "has_bearer": has_bearer,
                "is_valid": is_valid,
            }
        )