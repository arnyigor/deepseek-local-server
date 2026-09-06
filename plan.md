# Plan: deepseek-local-server (persistent service + MCP, DeepThink/expert mode)

Goal: turn this repo from a one-shot probe script into a long-running OpenAI-compatible
local server for chat.deepseek.com, mirroring `G:\Android\OpenideProjects\qwen-local-server`
architecture 1:1, plus an MCP tool (`ask_deepseek`) that forces DeepSeek's "DeepThink" (R1
reasoning / "expert") mode on before sending.

Source of truth for DOM selectors: probe runs already captured in this repo/session —
`probe_send2.log` (successful round trip, prompt "PROBE_OK_2"/"PROBE_OK_3") and the saved
dumps in `C:\Users\ArnyPC\AppData\Local\deepseek-probe\debug\probe-after-send-20260906-102327.html`.
Do not re-guess selectors already confirmed below; only re-probe if DeepSeek's DOM has
visibly changed (run `python deepseek_probe.py --send "..."` again to refresh dumps).

## Confirmed DOM facts (from the probe dumps, do not re-derive)

- Composer: `textarea[placeholder="Message DeepSeek"]` — plain `<textarea>`, matches qwen's
  `COMPOSER_SELECTORS` pattern already (`textarea:not([disabled])` works as-is).
- Send button: NOT a real `<button>` — it's a `<div class="ds-button ds-button--primary
  ds-button--filled ds-button--ci">` sitting at the composer's bottom-right corner
  (`rect≈{x:1192,y:551,w:34,h:34}` vs composer `rect≈{x:464,y:479,w:774,h:60}`). No
  aria-label/title/text at all. **Enter already works and was proven end-to-end** (see
  `probe_send2.log:34`: `--- Sent via keyboard shortcut: Enter`) — so submission should try
  `div.ds-button--primary` click first (best-effort, mirrors qwen), then fall back to
  `Enter` / `Control+Enter`, same order as qwen's `_submit_prompt`.
- Assistant response container: `div.ds-markdown.ds-assistant-message-main-content`
  (inside a wrapper `div.ds-message`). This is the reliable "answer text" selector —
  semantic class name, not a hashed one.
- User message container: `div.ds-message` whose child is `div.fbb737a4 > div.ds-collapsible-text`
  (the `fbb737a4` token is a hashed/build class like qwen's — treat as unstable). For the
  "did it submit" baseline-count check, don't depend on a precise user selector; just count
  `div.ds-message` elements total before/after (cheaper and doesn't rely on the hashed class).
- New chat button: not captured in the probe dumps — add best-effort selectors
  (`button[aria-label*="New chat" i]`, `button:has-text("New chat")`) same as qwen, wrapped
  in the same try/ignore-failure pattern; it's a non-critical convenience action.
- **DeepThink toggle ("expert" mode)** — this is the key new piece qwen doesn't have:
  ```html
  <div tabindex="0" aria-pressed="false" class="f79352dc ds-toggle-button ds-toggle-button--m">
    ...svg icon...
    <span class="_6dbc175">DeepThink</span>
  </div>
  ```
  Toggle state lives in `aria-pressed="true"/"false"` on the `div.ds-toggle-button` itself.
  There is a sibling toggle for `Search` with the identical structure/class, just different
  `<span>` text — ignore it (not requested, YAGNI).
  Locator strategy: find `div.ds-toggle-button:has(span:text-is("DeepThink"))`, read
  `aria-pressed`, click it if not `"true"`, then re-read `aria-pressed` to confirm (don't
  trust the click blindly — DeepSeek could rate-limit toggle clicks or already be maxed
  out on reasoning mode).
- Stop/generating indicator selector: **not captured** (no mid-generation dump was saved).
  Do not invent a selector for it. Port qwen's `_wait_for_response` stability-polling logic
  as-is (poll every `poll_interval_seconds`, require `stable_seconds` of unchanged text) —
  this is exactly what the probe script already relies on and it worked
  (`PROBE_OK_2`/`PROBE_OK_3` round-tripped fine with zero stop-button knowledge). Treat a
  real stop-button selector as a later optimization, not a blocker.
  `ponytail: no STOP_SELECTORS for DeepSeek yet — falls back to pure text-stability polling
  (slower to detect end-of-generation than qwen's stop-button check); add STOP_SELECTORS
  once a mid-generation HTML dump confirms the real markup.`

## Target layout

```
deepseek-local-server/
  pyproject.toml
  src/deepseek_local_server/
    __init__.py          (__version__ = "0.1.0")
    __main__.py           -> from deepseek_local_server.cli import main; main()
    config.py
    auth.py
    errors.py
    cli.py
    mcp_server.py
    service.py
    pi_config.py          (SKIP for v1 — see "Explicitly out of scope")
    browser/
      __init__.py
      manager.py
      dom.py
      worker.py
    openai/
      __init__.py
      content.py
      prompt.py
      responses.py
      schemas.py
      tool_protocol.py
    api/
      __init__.py
      app.py
      dependencies.py
  tests/
    test_config.py
    test_content.py
    test_tool_protocol.py
    test_browser_manager.py
    test_dom.py            (new — see Testing section)
  deepseek_probe.py         (KEEP as-is, useful for re-probing DOM later)
```

## Step-by-step

### 1. Scaffolding
1. Create `pyproject.toml` — copy `qwen-local-server/pyproject.toml`, replace all
   `qwen-local-server`/`qwen_local_server` with `deepseek-local-server`/`deepseek_local_server`,
   version `0.1.0`, description "OpenAI-compatible local server backed by DeepSeek Web through
   Playwright". Same deps (`fastapi`, `uvicorn[standard]`, `playwright`, `pydantic`, `httpx`,
   `mcp`, dev: `pytest`).
2. Create `src/deepseek_local_server/__init__.py` with `__version__ = "0.1.0"`.
3. Create `src/deepseek_local_server/__main__.py` mirroring qwen's.

### 2. Config (`config.py`)
Copy `qwen_local_server/config.py`, adapt field-by-field:
- Env prefix `QWEN_LOCAL_SERVER_*` → `DEEPSEEK_LOCAL_SERVER_*`.
- `qwen_url` → `deepseek_url`, default `"https://chat.deepseek.com/"`.
- `qwen_preferred_model: str = "Qwen3.8-Max"` → replace with
  `deepthink_enabled: bool = True` (env `DEEPSEEK_LOCAL_SERVER_DEEPTHINK`, parsed with the
  existing `_env_bool` helper) — this is the "always pick the expert model" switch the user
  asked for. Default `True` so DeepThink is forced on unless explicitly disabled.
- `model_id` default `"deepseek-web"`, `model_name` default `"DeepSeek Web (DeepThink, local adapter)"`.
- `_default_home()` → `LOCALAPPDATA/deepseek-local-server` (or `XDG_DATA_HOME` / `~/.local/share` fallbacks, unchanged logic).
- Keep `context_window`, `max_output_tokens`, `block_heavy_resources`, `max_prompt_chars`,
  `stable_seconds`, `poll_interval_seconds`, `request_timeout_seconds`, `port` (pick a
  different default port so it can run alongside qwen-local-server, e.g. `9874`), `validate()`
  unchanged.

### 3. `auth.py`, `errors.py`
Copy verbatim, only rename the import path (`qwen_local_server` → `deepseek_local_server`) and
the one user-facing message in `auth.py` (`Run \`qwen-local-server init\`` →
`` Run `deepseek-local-server init` ``). `errors.py`: rename `QwenPageError` →
`DeepSeekPageError` (keep `QwenLocalServerError`-equivalent base renamed to
`DeepSeekLocalServerError`, `AuthenticationRequiredError`, `BrowserProtocolError`,
`UnsupportedContentError`, `ToolProtocolError` unchanged in shape).

### 4. `browser/manager.py`
Copy **verbatim**, only change the import of `Settings` and the module's own package name.
No DeepSeek-specific logic lives here — this is the exact class discussed in the earlier
conversation (`start()` reuses the last live page across calls; `get_live_page`,
`new_page`, `close`, `reset` unchanged). This is what makes the service keep one window
open across requests instead of spawning a new Chromium each time, like `deepseek_probe.py`
currently does.

### 5. `browser/dom.py`
New content (not a copy) — use the confirmed selectors above:
```python
ASSISTANT_SELECTORS = ['.ds-markdown.ds-assistant-message-main-content']
USER_MESSAGE_COUNT_SELECTOR = '.ds-message'   # baseline count only, see note above
COMPOSER_SELECTORS = ['textarea[placeholder="Message DeepSeek"]', 'textarea:not([disabled])']
SEND_SELECTORS = ['div.ds-button--primary']   # best-effort; Enter fallback is primary path
NEW_CHAT_SELECTORS = ['button[aria-label*="New chat" i]', 'button:has-text("New chat")']
STOP_SELECTORS = []   # unknown — see plan notes, do not fabricate
DEEPTHINK_TOGGLE_SELECTOR = 'div.ds-toggle-button:has(span:text-is("DeepThink"))'
```
Port `SNAPSHOT_MESSAGES_JS`, `PAGE_ERROR_JS`, `is_junk_snapshot_text`, `clean_assistant_text`
from qwen's `dom.py` almost unchanged (the JS is generic DOM-walking, not Qwen-specific) —
just retarget `PAGE_ERROR_JS`'s auth/error phrase lists to what DeepSeek actually shows
(unknown yet; keep qwen's generic phrases like "login expired"/"network error" as a
reasonable starting point, refine later from real error dumps). Drop
`MODEL_TRIGGER_TEXT_PATTERN` and `PREFER_RESPONSE_SELECTORS` (Qwen-only A/B response-variant
picker; nothing in the DeepSeek dumps suggests this exists — YAGNI, don't port dead code).

### 6. `browser/worker.py`
Copy `qwen_local_server/browser/worker.py` structure, with these behavioral changes:
- Rename `QwenBrowserWorker` → `DeepSeekBrowserWorker`, `qwen_url` → `deepseek_url`.
- Delete `_select_preferred_model` (text-dropdown logic doesn't apply) and replace its call
  site in `query()` with a new `_ensure_deepthink(page)`:
  ```python
  async def _ensure_deepthink(self, page: Page) -> None:
      if not self._settings.deepthink_enabled:
          return
      try:
          toggle = page.locator(DEEPTHINK_TOGGLE_SELECTOR).first
          if not await toggle.count():
              return
          if (await toggle.get_attribute("aria-pressed")) == "true":
              return
          await toggle.click(timeout=3_000)
          await asyncio.sleep(0.3)
          # best-effort confirmation; don't fail the request over UI lag
          state = await toggle.get_attribute("aria-pressed")
          if state != "true":
              LOGGER.info("DeepThink toggle did not confirm as pressed (state=%r)", state)
      except Exception:
          LOGGER.info("could not enable DeepThink toggle", exc_info=True)
  ```
  Same "never break the send flow over a preference" philosophy as qwen's model-switch code.
- Delete `_keep_first_response_variant` and its call in `_wait_for_response` (no evidence
  DeepSeek has this variant-picker; qwen-only feature, YAGNI).
- `_submit_prompt`: try `SEND_SELECTORS` div click first, then `Control+Enter`/`Enter` —
  same order/logic as qwen, since Enter is already confirmed working from the probe.
- `_generation_running`: with empty `STOP_SELECTORS`, this always returns `False` — that's
  fine, `_wait_for_response`'s stability-timer already gates on it being `False` (i.e. it
  degrades gracefully to pure stability-polling, matching current probe behavior exactly).
- Everything else (`_find_composer`, `_fill_composer`, `_composer_text`,
  `_wait_until_submitted`, `_message_snapshot`, `_wait_for_response`, `_capture_debug`,
  `login_interactively`) ports with only naming/selector-source changes, no logic changes.

### 7. `openai/*` (schemas, content, prompt, responses, tool_protocol)
Copy near-verbatim; the only DeepSeek-specific edits are string constants, not logic:
- `tool_protocol.py`: `<<<QWEN_TOOL_CALL>>>` → `<<<DEEPSEEK_TOOL_CALL>>>` (and `_END_` pair),
  error message `f"Qwen attempted unknown tool: {name}"` → `f"DeepSeek attempted unknown tool: {name}"`.
- `prompt.py`: same tag constants updated to match tool_protocol.py; comment about "Qwen
  still sees it" → "DeepSeek still sees it".
- `responses.py`: `"system_fingerprint": "qwen-web-playwright-adapter"` →
  `"deepseek-web-playwright-adapter"`; comment about token estimate wording updated.
- `content.py`: error message "first Qwen Web adapter version" → "first DeepSeek Web adapter version".
- `schemas.py`: `owned_by: str = "qwen-local-server"` → `"deepseek-local-server"`.
- Import paths: `qwen_local_server.*` → `deepseek_local_server.*` throughout.

### 8. `api/dependencies.py`
Copy verbatim (bearer-token check has zero Qwen-specific logic).

### 9. `api/app.py`
Copy `qwen_local_server/api/app.py`, rename imports and exception types
(`QwenPageError`→`DeepSeekPageError`), title `"DeepSeek Local Server"`, error code
`"qwen_login_required"` → `"deepseek_login_required"`, `"qwen_web_error"` →
`"deepseek_web_error"`, header names `X-Qwen-Local-*` → `X-DeepSeek-Local-*`. Route
structure, health/models/chat-completions endpoints unchanged.

### 10. `service.py`
Copy `qwen_local_server/service.py` verbatim except: `QwenBrowserWorker` →
`DeepSeekBrowserWorker`, `LOGGER` name, and the model-id mismatch error message wording.
Session-continuation logic (`_continuation_delta`, single in-flight `_lock`, keep-page-open
reuse across turns of the same conversation) is generic — no changes needed.

### 11. `cli.py`
Copy `qwen_local_server/cli.py`, rename types/imports, `prog="deepseek-local-server"`,
help text "Expose DeepSeek Web as an OpenAI-compatible local model server". Keep
`init`/`auth`/`serve`/`doctor`/`chat`/`mcp` subcommands. **Drop** `install-pi` subcommand
and the `pi_config` import — see "Explicitly out of scope" below (nothing in this request
mentioned Pi integration; adding it would be scope creep neither asked for nor verifiable
against DeepSeek's own DOM).

### 12. `mcp_server.py`
Copy `qwen_local_server/mcp_server.py`, adapt:
- `server = MCPServer(name="deepseek-web", instructions=...)` — instructions text updated to
  mention DeepThink: *"Ask a plain-text question to DeepSeek (chat.deepseek.com) through a
  local browser bridge, with DeepThink (R1 reasoning / expert mode) enabled by default. ..."*
- Tool renamed `ask_qwen` → `ask_deepseek`, same signature
  `(question: str, ctx: Context, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS)`.
- All error/status strings updated (`"deepseek-local-server is not initialized"`,
  `` "Start it first with `deepseek-local-server serve`." ``, etc).
- This is the "MCP copy just for DeepSeek" the user asked for — it's a separate MCP server
  process/tool name (`ask_deepseek`) from qwen's `ask_qwen`, both can run side by side since
  they use different ports (see step 2) and different token/profile homes.

### 13. Explicitly out of scope for this pass (say so, don't silently build it)
- `pi_config.py` / `install-pi` CLI command — not requested, and Pi's `models.json` schema
  assumptions were validated against qwen, not verified for a second local provider entry;
  skip until asked.
- `Search` toggle (DeepSeek's web-search mode, sibling of DeepThink) — not requested, same
  toggle-button pattern would apply if ever needed (`ds-toggle-button:has(span:text-is("Search"))`).
- Streaming: keep qwen's "buffered" approach (`buffered_stream`/`completion_payload` as-is) —
  DeepSeek's SSE-in-browser output isn't observed token-by-token by this architecture either
  way (it's a DOM poll, not a real token stream), so there's nothing to change here.

### 14. Testing (ponytail: one runnable check per non-trivial branch, no more)
- `tests/test_config.py`: port qwen's env-var/validate tests with new prefix and defaults
  (especially assert `deepthink_enabled` defaults to `True` and reads `DEEPSEEK_LOCAL_SERVER_DEEPTHINK`).
- `tests/test_content.py`, `tests/test_tool_protocol.py`: port verbatim with renamed tags/imports
  (pure string/parsing logic, DeepSeek-agnostic).
- `tests/test_browser_manager.py`: port verbatim (`BrowserManager._live_pages` static logic,
  identical class).
- `tests/test_dom.py` (new): a small `is_junk_snapshot_text`/`clean_assistant_text` unit test,
  plus one assertion that `DEEPTHINK_TOGGLE_SELECTOR` is a non-empty, syntactically valid
  Playwright selector string (can't unit-test real DOM matching without a live page — that's
  what re-running `deepseek_probe.py` is for). This is the minimum "fails if the logic
  breaks" check for the one genuinely new piece of logic (DeepThink toggle selection).

### 15. Manual verification (do this before calling it done)
1. `pip install -e .[dev]` inside `deepseek-local-server` (own venv or reuse `.venv`).
2. `deepseek-local-server init` → creates home dir + token under `%LOCALAPPDATA%\deepseek-local-server`.
3. `deepseek-local-server auth` → opens **one** Chromium window (persistent profile), confirms
   already-logged-in session is picked up (profile dir is separate from `deepseek-probe`'s,
   so first run will need a fresh login — note this to the user, don't silently reuse the
   probe's profile dir since `Settings.profile_dir` differs by design from `deepseek_probe.py`'s
   `PROFILE_DIR`).
4. `deepseek-local-server serve` → then in a second terminal:
   `deepseek-local-server chat "Reply with exactly: SERVICE_OK"` — confirm: (a) only one
   Chromium window ever appears (not a new one per request — this was the original ask),
   (b) response contains `SERVICE_OK`, (c) DeepThink toggle screenshot/debug dump (on
   failure, `debug_dir`) shows `aria-pressed="true"` if you deliberately break the selector
   to force a debug capture.
5. `deepseek-local-server mcp` wired into an MCP client, call `ask_deepseek` once.

## Order to implement (checklist)

- [x] 1. Scaffolding (pyproject.toml, `__init__.py`, `__main__.py`)
- [x] 2. `config.py`
- [x] 3. `auth.py`, `errors.py`
- [x] 4. `browser/manager.py` (verbatim port)
- [x] 5. `browser/dom.py` (new, selectors above)
- [x] 6. `browser/worker.py` (ported + DeepThink toggle logic)
- [x] 7. `openai/*` (verbatim + string renames)
- [x] 8. `api/dependencies.py`, `api/app.py`
- [x] 9. `service.py`
- [x] 10. `cli.py`
- [x] 11. `mcp_server.py`
- [x] 12. Tests (section 14)
- [ ] 13. Manual verification (section 15)
