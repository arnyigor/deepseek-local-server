# deepseek-local-server v0.2 — hybrid direct + browser fallback

Local DeepSeek Web gateway for agents and OpenAI-compatible clients.

## What changed from v0.1

v0.1 drove the DeepSeek UI through Playwright for every request. v0.2 uses the internal Web Chat HTTP/SSE contract as the **primary transport** and keeps Playwright only for:

1. interactive login/auth capture;
2. emergency fallback when the direct contract fails.

Normal request path:

```text
Pi / Claude / OpenAI client
        |
        v
127.0.0.1:9874
        |
        v
Hybrid CompletionService
        |
        +--> Direct DeepSeek Web API + PoW + true SSE  [PRIMARY]
        |
        `--> Playwright UI automation                 [FALLBACK]
```

## Features

- OpenAI `POST /v1/chat/completions`
- true streaming for direct requests
- `reasoning_content` for thinking modes
- Fast, reasoning, search, Expert, Expert+reasoning aliases
- per-agent remote DeepSeek sessions (`x-agent-session` / `user`)
- automatic session reset when the client history changes
- basic OpenAI tool-call adapter
- MCP `ask_deepseek`
- basic Anthropic `/v1/messages` shim
- basic OpenAI Responses `/v1/responses` shim
- Playwright emergency fallback
- local bearer token is always required
- loopback-only binding is enforced

## Models

| Model | Web mode | Reasoning | Search |
|---|---|---:|---:|
| `deepseek-chat` | default | no | no |
| `deepseek-reasoner` | default | yes | no |
| `deepseek-chat-search` | default | no | yes |
| `deepseek-reasoner-search` | default | yes | yes |
| `deepseek-expert` | expert | no | no |
| `deepseek-v4-pro` | expert | yes | no |

Compatibility aliases: `deepseek-web -> deepseek-v4-pro`, `deepseek-r1 -> deepseek-reasoner`.

## Windows installation

```powershell
cd G:\path\to\deepseek-local-server-v2
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\pip.exe install -e ".[dev]"
```

No `playwright install` needed: `auth` and the browser fallback drive your real, already-installed
Google Chrome over the DevTools protocol instead of a Playwright-managed browser build. Set
`DEEPSEEK_LOCAL_SERVER_CHROME_PATH` if Chrome isn't in one of the default install locations.

## First setup

```powershell
.\.venv\Scripts\deepseek-local-server.exe init
.\.venv\Scripts\deepseek-local-server.exe auth
```

`auth` opens Chrome. Log in to DeepSeek and send one tiny message, for example `AUTH_OK`. The program captures the auth token, cookie, `x-hif-*` headers and SHA3 PoW WASM URL from **your own browser request**, stores them in:

```text
%LOCALAPPDATA%\deepseek-local-server\deepseek-auth.json
```

Do not commit or share that file.

Then:

```powershell
.\.venv\Scripts\deepseek-local-server.exe doctor
.\.venv\Scripts\deepseek-local-server.exe serve
```

In another terminal:

```powershell
.\.venv\Scripts\deepseek-local-server.exe chat "Reply exactly SERVICE_OK" --model deepseek-v4-pro
```

## OpenAI client

```python
from openai import OpenAI
from pathlib import Path
import os

home = Path(os.environ["LOCALAPPDATA"]) / "deepseek-local-server"
token = (home / "token").read_text().strip()

client = OpenAI(base_url="http://127.0.0.1:9874/v1", api_key=token)

stream = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{"role": "user", "content": "Analyze this design."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## MCP / Pi

`~/.pi/agent/mcp.json`:

```json
{
  "deepseek-web": {
    "command": "G:/path/to/deepseek-local-server-v2/.venv/Scripts/python.exe",
    "args": ["-m", "deepseek_local_server", "mcp"],
    "cwd": "G:/path/to/deepseek-local-server-v2",
    "lifecycle": "keep-alive",
    "directTools": true,
    "requestTimeoutMs": 600000
  }
}
```

MCP tool:

```text
ask_deepseek(
  question,
  mode="expert" | "instant",
  reasoning=true,
  search=false,
  new_conversation=false,
  timeout_seconds=330
)
```

Current DeepSeek Web contract does not expose search for Expert, so `mode="expert", search=true` is rejected instead of silently changing modes.

## Browser fallback

Enabled by default:

```powershell
$env:DEEPSEEK_LOCAL_SERVER_BROWSER_FALLBACK="1"
```

If the direct API fails **before streaming has emitted data**, the request can fall back to Playwright. Once a live direct stream has emitted bytes, automatic fallback is intentionally disabled because mixing two answers would corrupt the stream.

Disable fallback while testing the direct backend:

```powershell
$env:DEEPSEEK_LOCAL_SERVER_BROWSER_FALLBACK="0"
```

Force old browser-only behavior:

```powershell
$env:DEEPSEEK_LOCAL_SERVER_DIRECT="0"
$env:DEEPSEEK_LOCAL_SERVER_BROWSER_FALLBACK="1"
```

## Diagnostics

```powershell
# validate files only
.\.venv\Scripts\deepseek-local-server.exe doctor --offline

# validate auth + call DeepSeek PoW challenge endpoint
.\.venv\Scripts\deepseek-local-server.exe doctor
```

Useful API endpoints:

```text
GET  /health
GET  /v1/models
GET  /v1/model-capabilities
GET  /v1/sessions
POST /v1/sessions/reset?agent=<id|all>
POST /v1/chat/completions
POST /v1/messages
POST /v1/responses
```

## Environment variables

| Variable | Default |
|---|---|
| `DEEPSEEK_LOCAL_SERVER_HOST` | `127.0.0.1` |
| `DEEPSEEK_LOCAL_SERVER_PORT` | `9874` |
| `DEEPSEEK_LOCAL_SERVER_DIRECT` | `1` |
| `DEEPSEEK_LOCAL_SERVER_BROWSER_FALLBACK` | `1` |
| `DEEPSEEK_LOCAL_SERVER_HEADLESS` | `1` |
| `DEEPSEEK_LOCAL_SERVER_TIMEOUT_SECONDS` | `300` |
| `DEEPSEEK_LOCAL_SERVER_FETCH_TIMEOUT_SECONDS` | `60` |
| `DEEPSEEK_LOCAL_SERVER_SESSION_TTL_SECONDS` | `7200` |
| `DEEPSEEK_LOCAL_SERVER_MAX_SESSION_MESSAGES` | `100` |
| `DEEPSEEK_LOCAL_SERVER_MAX_CONCURRENT_DIRECT` | `8` |
| `DEEPSEEK_LOCAL_SERVER_MAX_PROMPT_CHARS` | `200000` |

## Attribution

The direct Web API implementation was informed by the MIT-licensed `ForgetMeAI/FreeDeepseekAPI` project. See `THIRD_PARTY_NOTICES.md`.

## Important limitations

This is an experimental adapter for DeepSeek Web, not the official DeepSeek API. Internal endpoints, PoW WASM exports, request headers, stream patches or UI selectors can change without notice.

The direct backend is isolated under `direct/` specifically so those changes do not spread through the OpenAI/MCP layers.

The implementation does not bypass CAPTCHA, 2FA or login challenges. Re-run `auth` and complete those checks normally in the browser.
