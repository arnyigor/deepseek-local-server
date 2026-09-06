# deepseek-local-server

OpenAI-compatible local server backed by **chat.deepseek.com** through Playwright (Chromium),
with the **Expert model** and the **DeepThink (reasoning) toggle** enabled by default.

It drives the real DeepSeek Web UI in a persistent browser profile: you log in once, then any
OpenAI-compatible client (or the bundled MCP tool) can send chat requests and get DeepThink
answers back — no API key for the official API needed.

Architecture mirrors `qwen-local-server` — see [plan.md](plan.md) for port notes and confirmed DOM selectors.

## How it works

```
Client (OpenAI SDK / curl / MCP ask_deepseek)
        │  HTTP POST /v1/chat/completions  (Bearer token)
        ▼
deepseek-local-server serve   ← FastAPI on 127.0.0.1:9874
        │  serializes requests through a single browser worker
        ▼
Chromium (persistent profile, chat.deepseek.com)
        │  selects Expert model → enables DeepThink toggle → sends → polls until the
        │  answer is stable (stable_seconds) → extracts text
        ▼
Answer returned as a standard OpenAI chat completion
```

Key points:

- **One browser, one request at a time.** Requests are queued in the worker; the browser
  profile is launched lazily on the first request and reused afterwards.
- **Login is a persistent profile**, not stored cookies. The profile lives in
  `%LOCALAPPDATA%\deepseek-local-server\browser-profile` (Windows) — the same directory
  Chromium uses, so **only one process may use it at a time**. If an `auth` window is still
  open, `serve` cannot launch its own Chromium ("profile is already in use").
- **Expert model + DeepThink are on by default.** Before every send the worker selects the
  "Expert" option in the top-bar model picker (Instant/Expert) and turns the DeepThink toggle
  on (`DEEPSEEK_LOCAL_SERVER_DEEPTHINK=1`). Reasoning answers can take minutes — the default
  request timeout is 300 s.
- The server **only listens on loopback** and requires the local bearer token from `init`.

## Setup (one-time)

```powershell
# Windows PowerShell (note: PS 5.1 has no `&&` — use `;` or separate lines)
cd G:/Android/OpenideProjects/deepseek-local-server
pip install -e .[dev]
python -m playwright install chromium
```

## Usage

### 1. Init (creates home dir + API token)

```powershell
./.venv/Scripts/deepseek-local-server.exe init
```

Creates `%LOCALAPPDATA%\deepseek-local-server\` with `browser-profile/`, `token`, `debug/`.

### 2. Auth (log in once)

```powershell
./.venv/Scripts/deepseek-local-server.exe auth
```

Opens a Chromium window → sign in to DeepSeek → wait until the chat input is visible →
**press Enter in the terminal** (keep the window open until it closes). The profile is
saved automatically; you will not need to log in again.

### 3. Serve (keep running)

In a **separate terminal**, leave this running:

```powershell
./.venv/Scripts/deepseek-local-server.exe serve
```

Starts the OpenAI-compatible endpoint at `http://127.0.0.1:9874/v1`.

### 4. Verify

```powershell
./.venv/Scripts/deepseek-local-server.exe doctor
./.venv/Scripts/deepseek-local-server.exe chat "Reply with exactly: SERVICE_OK"
```

`doctor` prints health + `/v1/models`; `chat` sends a real request through the browser.
A healthy end-to-end setup returns `SERVICE_OK`.

### OpenAI-compatible API

```powershell
$token = Get-Content "$env:LOCALAPPDATA\deepseek-local-server\token"
curl http://127.0.0.1:9874/v1/chat/completions `
  -H "Authorization: Bearer $token" -H "Content-Type: application/json" `
  -d '{"model":"deepseek-web","messages":[{"role":"user","content":"hi"}],"stream":false}'
```

Any OpenAI SDK works with `base_url="http://127.0.0.1:9874/v1"` and `api_key=<token>`.

### MCP tool (`ask_deepseek`)

Runs the same endpoint as a thin stdio MCP wrapper (server name `deepseek-web`,
single tool `ask_deepseek(question, timeout_seconds, new_conversation=False, mode="expert")`). It does **not** start `serve`
itself — keep `serve` running separately.

Calls continue the current conversation by default. The MCP process keeps successful
questions and answers in memory and sends the history so the backend can reuse the
active browser chat. Set `new_conversation: true` to clear history and start a new chat:

```json
{"question": "Remember the code BLUE-CAT-42", "new_conversation": true}
{"question": "What code did I give you?"}
{"question": "Start a different topic", "new_conversation": true}
```

Keep the MCP process alive between calls (Pi: `lifecycle: "keep-alive"`). Restarting
or reconnecting the process clears its history. Concurrent calls are serialized;
failed calls are not added to history. An explicit reset clears history even if its
request fails. The backend has one active browser chat: if another API/MCP client
replaces it, the next call restores the conversation from history in a new browser chat.

Use `mode: "instant"` to select Instant without enabling Expert or DeepThink:

```json
{"question": "Give a short answer", "mode": "instant"}
{"question": "Continue", "mode": "instant"}
```

The mode applies to each call and defaults to `"expert"`; it does not reset chat history.
It can be combined with `new_conversation: true`. The HTTP `/v1/chat/completions`
endpoint accepts the same optional `mode` field. Restart `serve` and reconnect the
MCP process after updating so both ends recognize the parameter.

Pi (`~/.pi/agent/mcp.json`):

```json
"deepseek-web": {
  "command": "G:/Android/OpenideProjects/deepseek-local-server/.venv/Scripts/python.exe",
  "args": ["-m", "deepseek_local_server", "mcp"],
  "cwd": "G:/Android/OpenideProjects/deepseek-local-server",
  "lifecycle": "keep-alive",
  "directTools": true,
  "requestTimeoutMs": 600000
}
```

Claude Code / Desktop (`~/.claude.json` or `.mcp.json`):

```json
"deepseek-web": {
  "type": "stdio",
  "command": "G:/Android/OpenideProjects/deepseek-local-server/.venv/Scripts/deepseek-local-server.exe",
  "args": ["mcp"]
}
```

After editing the MCP config, restart the agent so it picks up the change.

## Tests

```powershell
./.venv/Scripts/py.test.exe -q     # 13 tests
```

## Configuration (environment variables)

All optional; defaults in parentheses.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DEEPSEEK_LOCAL_SERVER_HOME` | `%LOCALAPPDATA%\deepseek-local-server` | Home dir (profile, token, debug) |
| `DEEPSEEK_LOCAL_SERVER_HOST` | `127.0.0.1` | Bind address (loopback only, enforced) |
| `DEEPSEEK_LOCAL_SERVER_PORT` | `9874` | Port |
| `DEEPSEEK_LOCAL_SERVER_DEEPSEEK_URL` | `https://chat.deepseek.com/` | Target page |
| `DEEPSEEK_LOCAL_SERVER_HEADLESS` | `0` | Run Chromium headless (careful: login needs a visible window) |
| `DEEPSEEK_LOCAL_SERVER_TIMEOUT_SECONDS` | `300` | Per-request browser timeout |
| `DEEPSEEK_LOCAL_SERVER_STABLE_SECONDS` | `2.0` | Answer must be unchanged this long before it's accepted |
| `DEEPSEEK_LOCAL_SERVER_POLL_INTERVAL_SECONDS` | `0.35` | Poll interval |
| `DEEPSEEK_LOCAL_SERVER_MAX_PROMPT_CHARS` | `900000` | Prompt size cap |
| `DEEPSEEK_LOCAL_SERVER_MODEL_ID` | `deepseek-web` | Model id exposed via the API |
| `DEEPSEEK_LOCAL_SERVER_DEEPTHINK` | `1` | DeepThink (reasoning) toggle; Expert model is always selected |
| `DEEPSEEK_LOCAL_SERVER_BLOCK_HEAVY_RESOURCES` | `1` | Block heavy page resources for speed |

## Troubleshooting

- **`BrowserType.launch_persistent_context: Target ... closed` / "profile is already in use"**
  — another Chromium (usually the `auth` window) still holds the profile. Close it and retry.
- **`deepseek-local-server is not initialized`** (from `ask_deepseek`) — run `init`.
- **`Could not reach deepseek-local-server at http://127.0.0.1:9874`** (from `ask_deepseek`)
  — start `serve` in a separate terminal.
- **Timed out after ~300 s** — normal for long DeepThink answers; raise
  `DEEPSEEK_LOCAL_SERVER_TIMEOUT_SECONDS` or pass a larger `timeout_seconds` to `ask_deepseek`.
- **Login expired / page changed** — re-run `auth` and log in again.

## Project layout

```
src/deepseek_local_server/
  cli.py            # init / auth / serve / doctor / chat / mcp
  config.py         # Settings from env, validation, paths
  auth.py           # local API token
  service.py        # request queue + browser worker orchestration
  browser/          # Playwright manager, DOM selectors, worker (send/poll/parse)
  api/              # FastAPI app: /health, /v1/models, /v1/chat/completions
  openai/           # OpenAI schema/content helpers
  mcp_server.py     # ask_deepseek stdio MCP wrapper
tests/              # 13 unit tests (config, dom, content, tool protocol, browser manager)
```
