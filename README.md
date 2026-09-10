# deepseek-local-server v0.2 — direct DeepSeek Web API gateway

Local DeepSeek Web gateway for agents and OpenAI-compatible clients. All chat traffic goes through the direct DeepSeek Web HTTP/SSE contract (SHA3 PoW executed locally via wasmtime). Chrome is used only once, for interactive auth capture.

## What changed from v0.1

v0.1 drove the DeepSeek UI through Playwright for every request. v0.2 talks the internal Web Chat HTTP/SSE contract directly. The browser fallback and Playwright automation layer were removed entirely — Chrome over CDP is used only for the one-time `auth` capture from your own logged-in browser.

Request path:

```text
Pi / Claude / OpenAI client
        |
        v
127.0.0.1:9874
        |
        v
CompletionService --> Direct DeepSeek Web API + PoW + true SSE
```

## Features

- OpenAI `POST /v1/chat/completions` (streaming and non-streaming)
- true upstream SSE streaming, `reasoning_content` for thinking modes
- image input (vision) via OpenAI image content parts
- per-agent remote DeepSeek sessions (`x-agent-session` / `user`)
- automatic session reset when the client history changes
- basic OpenAI tool-call adapter
- MCP `ask_deepseek`: always reasoning + web search; the result is the answer (markdown and LaTeX rendered for plain-text hosts) while reasoning streams as a one-line status ticker; images passed by local file path
- basic Anthropic `/v1/messages` shim
- basic OpenAI Responses `/v1/responses` shim
- local bearer token is always required
- loopback-only binding is enforced

## Models

| Model | Reasoning | Search | Label |
|---|---:|---:|---|
| `deepseek-chat` | no | no | Fast |
| `deepseek-reasoner` | yes | no | Fast + reasoning |
| `deepseek-chat-search` | no | yes | Fast + web search |
| `deepseek-reasoner-search` | yes | yes | Fast + reasoning + web search |

Compatibility aliases (DeepSeek merged Instant/Expert/Vision into one model, 2026-09): `deepseek-web`, `deepseek-expert`, `deepseek-v4-pro` -> `deepseek-reasoner`; `deepseek-r1` -> `deepseek-reasoner`; `deepseek-r1-search` -> `deepseek-reasoner-search`; `deepseek-v3`, `deepseek-default` -> `deepseek-chat`.

## Windows installation

```powershell
cd G:\path\to\deepseek-local-server-v2
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\pip.exe install -e ".[dev]"
```

No `playwright install` needed: `auth` drives your real, already-installed Google Chrome over the DevTools protocol instead of a Playwright-managed browser build. Set `DEEPSEEK_LOCAL_SERVER_CHROME_PATH` if Chrome isn't in one of the default install locations.

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
.\.venv\Scripts\deepseek-local-server.exe chat "Reply exactly SERVICE_OK" --model deepseek-reasoner
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
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "Analyze this design."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.reasoning_content:
        print("[think]", delta.reasoning_content, end="", flush=True)
    else:
        print(delta.content or "", end="", flush=True)
```

Images: send OpenAI vision `content` parts with `image_url` entries whose `url` is a base64 data URL (`data:image/png;base64,...`) — bytes are uploaded to DeepSeek directly. In MCP, pass `image_path` with a local file path instead.

## MCP / Pi

`~/.pi/agent/mcp.json`:

```json
{
  "deepseek-web": {
    "command": "G:/path/to/deepseek-local-server-v2/.venv/Scripts/python.exe",
    "args": ["-m", "deepseek_local_server", "mcp"],
    "cwd": "G:/path/to/deepseek-local-server-v2",
    "lifecycle": "keep-alive",
    "directTools": false,
    "requestTimeoutMs": 600000
  }
}
```

MCP tool:

```text
ask_deepseek(
  question,
  new_conversation=false,  # clear MCP-side history before this call
  image_path=None,         # local image file, e.g. "C:/pics/photo.jpg" (vision)
  timeout_seconds=<request timeout + 30>
)
```

Reasoning and web search are always on (`deepseek-reasoner-search`); there are no mode toggles.

For live reasoning while the model thinks, the tool must be reachable through the proxy path, because pi's progress bridge only runs there — set `"directTools": false` for this server (pi then shows the running reasoning as a status ticker). With `directTools: true` pi drops progress notifications entirely, so nothing appears until the call finishes.

Result layout:

- the **answer comes first**, so a host that shows only the first lines of a collapsed tool block still shows the answer;
- while thinking, a compact one-line ticker (`thinking: …` + the newest ~140 chars) is pushed about every 1.5s — pi replaces the same status line in place;
- callers that cannot receive progress (direct-tool calls) get a **bounded** slice of the chain appended after the answer (`4k head + 2k tail` plus an `… [N chars of reasoning omitted] …` marker), dimmed with SGR 90.

Session history stores the clean answer only. Conversation history is kept process-wide and survives across calls; `new_conversation=true` resets it.

## Rendering for plain-text hosts

MCP hosts print tool output literally (pi even wraps it in its theme's `toolOutput` colour), so the answer is normalised before it is returned:

- markdown tables become box-drawing tables (alignment taken from the `---:` row, cells wrapped, width budget 98, CJK/emoji measured in terminal cells);
- markdown and LaTeX fences around the whole answer are unwrapped; ```` ```python ```` and other real code blocks stay verbatim;
- LaTeX is converted to readable Unicode: `\frac{a}{b}` → `a/b`, `\sqrt{…}` → `√(…)`, `\mu` → μ, `\cdot` → ·, `10^{24}` → `10²⁴`, `v_1` → `v₁`, `\dot{m}` → `ṁ`, `\ln`, `\SI{a}{b}`, `\section{…}`, `tabular` → box table, and so on;
- headings, `**bold**`, `[links](url)`, `>` markers and math delimiters are dropped; inline `$math$` is only unwrapped when it looks like math, so prices stay intact.

## Testing

The renderer is pinned by snapshots of **real** DeepSeek answers:

```powershell
.\.venv\Scripts\python.exe -m pytest          # unit + snapshot suite
.\.venv\Scripts\python.exe capture_fixtures.py  # capture new real answers (needs the gateway)
.\.venv\Scripts\python.exe refresh_fixtures.py  # re-render the saved fixtures offline
.\.venv\Scripts\python.exe probe_reasoning_live.py   # live ticker check
.\.venv\Scripts\python.exe probe_reasoning_block.py  # live no-ticker layout check
```

`tests/fixtures/` keeps each answer as `<name>.raw.txt` (as the model returned it) next to `<name>.expected.txt` (what the renderer must produce). Review the diff before committing a refresh.

## Rendering in pi

Every line of MCP tool output is painted with the theme's `toolOutput` colour (`gray` in the shipped dark theme); the reasoning block overrides that per line with SGR 90, so it stays grey, and the answer keeps the theme colour. To make the answer white, copy `dark.json` from the pi install into `<agent-dir>/themes/`, set `colors.toolOutput` to `text`, and pick that theme with `/theme`. A collapsed tool block shows only its first lines (`collapsedResultLines`, 1–3 in `mcp.json` settings); press Ctrl+O or start pi with `--verbose` to see the whole result.

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
| `DEEPSEEK_LOCAL_SERVER_DEEPSEEK_URL` | `https://chat.deepseek.com/` |
| `DEEPSEEK_LOCAL_SERVER_CHROME_PATH` | auto-detect |
| `DEEPSEEK_LOCAL_SERVER_CHROME_DEBUG_PORT` | `9333` |
| `DEEPSEEK_LOCAL_SERVER_TIMEOUT_SECONDS` | `300` |
| `DEEPSEEK_LOCAL_SERVER_FETCH_TIMEOUT_SECONDS` | `60` |
| `DEEPSEEK_LOCAL_SERVER_SESSION_TTL_SECONDS` | `7200` |
| `DEEPSEEK_LOCAL_SERVER_MAX_SESSION_MESSAGES` | `100` |
| `DEEPSEEK_LOCAL_SERVER_MAX_CONCURRENT_DIRECT` | `8` |
| `DEEPSEEK_LOCAL_SERVER_MAX_PROMPT_CHARS` | `200000` |

## Attribution

The direct Web API implementation was informed by the MIT-licensed `ForgetMeAI/FreeDeepseekAPI` project. See `THIRD_PARTY_NOTICES.md`.

## Important limitations

This is an experimental adapter for DeepSeek Web, not the official DeepSeek API. Internal endpoints, PoW WASM exports, request headers, stream patches or auth headers can change without notice.

The direct backend is isolated under `direct/` specifically so those changes do not spread through the OpenAI/MCP layers.

The implementation does not bypass CAPTCHA, 2FA or login challenges. Re-run `auth` and complete those checks normally in the browser.
