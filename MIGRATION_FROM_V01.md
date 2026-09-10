# Migration from deepseek-local-server v0.1

The public goal stays the same, but v0.2 changes the transport hierarchy:

`client -> local API/MCP -> Direct Web backend -> DeepSeek`

The v0.1 browser automation layer is gone: there is no request-time Playwright fallback anymore, and `stream=true` no longer exists as a buffered browser polling path.

## What changes

- Run `deepseek-local-server auth` again. v0.2 captures the Web bearer token, HIF headers, cookies, client/app versions, user-agent and SHA3 WASM URL required by the direct transport.
- Keep using `deepseek-local-server serve` and the OpenAI-compatible `/v1/chat/completions` endpoint.
- MCP remains available through `deepseek-local-server mcp` and exposes `ask_deepseek`.
- `stream=true` is now a real upstream SSE stream for normal text requests instead of buffered browser polling.
- Use model aliases rather than relying on UI labels: `deepseek-chat`, `deepseek-reasoner`, `deepseek-chat-search`, `deepseek-reasoner-search`, `deepseek-expert`, `deepseek-v4-pro`.
- Browser fallback was removed entirely: every request goes through the direct Web backend. If a direct call fails, the error surfaces as-is instead of falling back to UI automation.
- `auth` drives your real, installed Google Chrome over the DevTools protocol (`connect_over_cdp`) instead of launching a Playwright-managed browser build. No `playwright install` step. Set `DEEPSEEK_LOCAL_SERVER_CHROME_PATH` if Chrome isn't in a default install location, or `DEEPSEEK_LOCAL_SERVER_CHROME_DEBUG_PORT` to change the debug port (default `9333`).

## Recommended migration check

```powershell
pip install -e .
deepseek-local-server init
deepseek-local-server auth
deepseek-local-server doctor
deepseek-local-server serve
```

Then in another terminal:

```powershell
curl.exe http://127.0.0.1:8765/v1/chat/completions `
  -H "Authorization: Bearer YOUR_LOCAL_TOKEN" `
  -H "Content-Type: application/json" `
  -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"Return exactly: OK"}],"stream":false}'
```

Do not expose this server to the LAN or Internet without adding an explicit deployment security layer. The default design is loopback-only.
