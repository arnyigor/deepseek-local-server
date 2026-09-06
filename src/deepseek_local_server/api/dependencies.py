from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request, status


async def require_bearer_token(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    expected = request.app.state.api_token
    prefix = "Bearer "
    supplied = authorization[len(prefix):].strip() if authorization and authorization.startswith(prefix) else ""
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid or missing bearer token",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )
