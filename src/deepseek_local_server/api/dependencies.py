from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status


async def require_bearer_token(request: Request) -> None:
    expected = request.app.state.api_token
    header = request.headers.get("authorization", "")
    supplied = header[7:] if header.lower().startswith("bearer ") else ""
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing bearer token")
