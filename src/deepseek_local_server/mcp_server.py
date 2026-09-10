from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
import time
import unicodedata
from pathlib import Path
from uuid import uuid4

import httpx
from mcp.server.mcpserver import Context, MCPServer

from deepseek_local_server.auth import read_api_token
from deepseek_local_server.config import Settings

_settings = Settings.from_env()
_DEFAULT_TIMEOUT = _settings.request_timeout_seconds + 30
_history: list[dict[str, str]] = []
_lock = asyncio.Lock()
_session_id = f"mcp-{uuid4().hex}"

server = MCPServer(
    name="deepseek-web",
    instructions=(
        "Use DeepSeek Web as a second-opinion/research model through a local gateway. "
        "All requests go through the direct DeepSeek Web API. "
        "DeepSeek merged its Instant/Expert/Vision modes into a single model (2026-09); "
        "reasoning and web search are always on and the full reasoning is returned with the answer. "
        "The merged model natively understands images: pass image_path to a local "
        "image file to ask about it. Pass all required context in the question."
    ),
)


MODEL = "deepseek-reasoner-search"  # reasoning + web search are always on

# Live reasoning ticker: one short single line, so it never floods the transcript.
NOTIFY_INTERVAL_SECONDS = 1.5
NOTIFY_TAIL_CHARS = 140

# The chain stored in the result must stay small: hosts truncate oversized tool
# output from the head, which would eat the answer if the chain came first.
RESULT_REASONING_HEAD_CHARS = 4_000
RESULT_REASONING_TAIL_CHARS = 2_000

# Long cells are wrapped inside the table so the box stays readable in a narrow TUI.
TABLE_MAX_COLUMN_CHARS = 34
TABLE_MAX_WIDTH_CHARS = 98


def _can_receive_progress(ctx: Context | None) -> bool:
    """True when the request carried a progress token (i.e. notifications land somewhere).

    Hosts only inject it for proxy-style calls; direct tool calls get nothing, so
    there the result alone carries the reasoning.
    """
    if ctx is None:
        return False
    try:
        meta = ctx.request_context.meta
    except Exception:
        return False
    # the framework normalises _meta.progressToken into meta["progress_token"]
    return bool(isinstance(meta, dict) and (meta.get("progress_token") or meta.get("progressToken")))


_GRAY = "\x1b[90m"
_RESET_FG = "\x1b[39m"


def _trim_reasoning(text: str) -> str:
    """Bound the chain kept in the result so the answer can never be pushed out."""
    if len(text) <= RESULT_REASONING_HEAD_CHARS + RESULT_REASONING_TAIL_CHARS:
        return text
    omitted = len(text) - RESULT_REASONING_HEAD_CHARS - RESULT_REASONING_TAIL_CHARS
    head = text[:RESULT_REASONING_HEAD_CHARS]
    tail = text[-RESULT_REASONING_TAIL_CHARS:]
    return f"{head}\n… [{omitted} chars of reasoning omitted] …\n{tail}"


def _dim(text: str) -> str:
    """Paint text grey inside a terminal line.

    MCP gives no styling channel, and hosts paint every result line with their
    own "tool output" colour, so the reasoning is dimmed with SGR codes that win
    over the outer colour; the final reset must not leak into the answer.
    """
    return "\n".join(f"{_GRAY}{line}{_RESET_FG}" if line else line for line in text.split("\n"))


def _char_width(char: str) -> int:
    """Terminal cells occupied by one character (CJK and emoji take two)."""
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _display_width(text: str) -> int:
    return sum(_char_width(char) for char in text)


def _split_row(line: str) -> list[str]:
    """Cells of a markdown table row; handles escaped pipes and drops bold markers."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    cells: list[str] = []
    current = ""
    index = 0
    while index < len(body):
        char = body[index]
        next_char = body[index + 1] if index + 1 < len(body) else ""
        if char == "\\" and next_char in {"|", "\\"}:
            current += next_char  # escaped pipe or escaped backslash
            index += 2
            continue
        if char == "|":
            cells.append(current)
            current = ""
        else:
            current += char
        index += 1
    cells.append(current)
    return [_strip_markers(cell.strip()) for cell in cells]


def _is_table_separator(line: str) -> bool:
    cells = _split_row(line)
    return bool(cells) and all(set(cell) <= set("-: ") and "-" in cell for cell in cells)


def _alignment(cell: str) -> str:
    left, right = cell.startswith(":"), cell.endswith(":")
    if left and right:
        return "center"
    if right:
        return "right"
    return "left"


def _take_width(text: str, limit: int) -> tuple[str, str]:
    """Split text so the head fits the limit in terminal cells."""
    used = 0
    for index, char in enumerate(text):
        width = _char_width(char)
        if used + width > limit:
            return text[:index], text[index:]
        used += width
    return text, ""


def _wrap_cell(text: str, limit: int) -> list[str]:
    """Greedy word wrap measured in terminal cells."""
    if _display_width(text) <= limit:
        return [text]
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if _display_width(candidate) <= limit:
            current = candidate
            continue
        if current:
            lines.append(current)
        while _display_width(word) > limit:
            piece, word = _take_width(word, limit)
            lines.append(piece)
        current = word
    if current or not lines:
        lines.append(current)
    return lines


def _box_table(rows: list[str]) -> list[str]:
    """Draw a markdown table as box-drawing art; hosts render results as plain text."""
    header = _split_row(rows[0])
    aligns = [_alignment(cell) for cell in _split_row(rows[1])]
    body = [_split_row(row) for row in rows[2:]]

    def cell(row: list[str], index: int) -> str:
        return row[index] if index < len(row) else ""

    columns = max(len(header), 1, *(len(row) for row in body)) if body else max(len(header), 1)
    aligns += ["left"] * (columns - len(aligns))
    widths = [
        max([_display_width(cell(header, i)), *(_display_width(cell(row, i)) for row in body)])
        for i in range(columns)
    ]
    widths = [min(width, TABLE_MAX_COLUMN_CHARS) for width in widths]

    def table_width() -> int:
        return sum(widths) + 3 * columns + 1

    # shrink the widest column until the whole box fits the budget
    while table_width() > TABLE_MAX_WIDTH_CHARS and max(widths) > 10:
        widths[widths.index(max(widths))] -= 1

    def wrap(values: list[str]) -> list[list[str]]:
        return [_wrap_cell(cell(values, i), widths[i]) for i in range(columns)]

    def pad(text: str, index: int) -> str:
        gap = max(0, widths[index] - _display_width(text))
        if aligns[index] == "right":
            return " " * gap + text
        if aligns[index] == "center":
            left = gap // 2
            return " " * left + text + " " * (gap - left)
        return text + " " * gap

    def rule(left: str, joint: str, right: str) -> str:
        return left + joint.join("─" * (width + 2) for width in widths) + right

    def row(values: list[str]) -> list[str]:
        wrapped = wrap(values)
        height = max(len(cell_lines) for cell_lines in wrapped)
        return [
            "│ "
            + " │ ".join(pad(cell_lines[k] if k < len(cell_lines) else "", i) for i, cell_lines in enumerate(wrapped))
            + " │"
            for k in range(height)
        ]

    lines = [rule("┌", "┬", "┐"), *row(header), rule("├", "┼", "┤")]
    for values in body:
        lines.extend(row(values))
    lines.append(rule("└", "┴", "┘"))
    return lines


def _clean_inline(text: str) -> str:
    """Strip markdown markers that hosts print literally, leaving code spans alone."""
    parts = re.split(r"(`[^`]*`)", text)
    return "".join(part if part.startswith("`") else _strip_markers(part) for part in parts)


LATEX_SYMBOLS = {
    "mu": "μ",
    "pi": "π",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "Delta": "Δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "rho": "ρ",
    "sigma": "σ",
    "Sigma": "Σ",
    "tau": "τ",
    "phi": "φ",
    "omega": "ω",
    "Omega": "Ω",
    "approx": "≈",
    "times": "×",
    "cdot": "·",
    "pm": "±",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "ne": "≠",
    "neq": "≠",
    "infty": "∞",
    "sum": "Σ",
    "prod": "Π",
    "int": "∫",
    "partial": "∂",
    "nabla": "∇",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "circ": "∘",
    "oplus": "⊕",
    "ominus": "⊖",
    "otimes": "⊗",
    "equiv": "≡",
    "propto": "∝",
    "sim": "∼",
    "angle": "∠",
    "cdots": "⋯",
    "ldots": "…",
    "prime": "′",
    "degree": "°",
    "ln": "ln",
    "log": "log",
    "lg": "lg",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "cot": "cot",
    "max": "max",
    "min": "min",
    "lim": "lim",
    "det": "det",
}
SUPERSCRIPT = str.maketrans("0123456789+-n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ")
SUBSCRIPT = str.maketrans("0123456789+-aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")
# accents combine with the base letter: \dot{m} -> ṁ
COMBINING = {"dot": "\u0307", "ddot": "\u0308", "bar": "\u0304", "hat": "\u0302", "tilde": "\u0303", "vec": "\u20d7"}
# only known commands are substituted, longest name first: \ln must still match in "\lnm"
_LATEX_COMMAND_RE = re.compile("\\\\(" + "|".join(sorted(map(re.escape, LATEX_SYMBOLS), key=len, reverse=True)) + ")")


def _script(body: str, table: dict[int, str]) -> str:
    if _is_scriptable(body, table):
        return body.lower().translate(table)
    if len(body) == 1:
        return body  # no Unicode form (e.g. subscript f): just attach the letter
    return f"({body})"


def _superscript(body: str) -> str:
    """Superscripts that Unicode cannot express keep the explicit form: e^x -> e^(-x)."""
    return body.lower().translate(SUPERSCRIPT) if _is_scriptable(body, SUPERSCRIPT) else f"({body})"


def _is_scriptable(body: str, table: dict[int, str]) -> bool:
    return bool(body) and all(ord(char) in table for char in body.lower())


def _render_math(text: str) -> str:
    """Turn common LaTeX into readable Unicode; unknown commands are kept verbatim."""
    for accent, mark in COMBINING.items():
        text = re.sub(rf"\\{accent}\s*\{{([^{{}}]*)\}}", lambda m, mark=mark: f"{m.group(1)}{mark}", text)
    for _ in range(4):  # innermost fractions first, one level of nested braces allowed
        # a fraction glued to a function name keeps the grouping: \ln\frac{a}{b} -> ln(a/b)
        text = re.sub(
            r"(?<=[A-Za-z0-9)])\\(?:d|t|c)?frac\s*\{((?:[^{}]|\{[^{}]*\})*)\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
            r"(\1/\2)",
            text,
        )
        text = re.sub(
            r"\\(?:d|t|c)?frac\s*\{((?:[^{}]|\{[^{}]*\})*)\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
            r"\1/\2",
            text,
        )
    text = re.sub(r"\\sqrt\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"√(\1)", text)
    text = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:left|right|displaystyle)\s*", "", text)
    text = text.replace("{,}", ",").replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    text = re.sub(r"\\qquad|\\quad|\\enspace", "   ", text)
    text = re.sub(r"\\hspace\s*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\\s", " ", text)  # \ followed by whitespace is a plain space
    text = re.sub(r"\^\{([^{}]*)\}", lambda m: _superscript(m.group(1)), text)
    text = re.sub(r"_\{([^{}]*)\}", lambda m: _script(m.group(1), SUBSCRIPT), text)
    text = re.sub(r"\^([0-9]+)(?![0-9])", lambda m: _superscript(m.group(1)), text)
    text = re.sub(
        r"(?<=[A-Za-z0-9)])([_^])([0-9a-z])(?![A-Za-z0-9])",
        lambda m: _script(m.group(2), SUBSCRIPT)
        if m.group(1) == "_"
        else _superscript(m.group(2)),
        text,
    )
    text = _LATEX_COMMAND_RE.sub(lambda m: LATEX_SYMBOLS[m.group(1)], text)
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", text))


def _strip_markers(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    # math delimiters are noise in plain text; the formula itself stays
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text)
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text)
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text)
    # inline $math$: only when it looks like math, so prices are left alone
    text = re.sub(r"\$([A-Za-z])\$", r"\1", text)
    text = re.sub(r"\$(?=[^$]*?[\\_^{}])([^$]+)\$", r"\1", text)
    return _render_math(text)


def _clean_line(line: str) -> str:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    heading = re.match(r"#{1,6}\s+(.*)$", stripped)
    if heading:
        return indent + heading.group(1).strip()
    quote = re.match(r">\s?(.*)$", stripped)
    if quote:
        return indent + _clean_inline(quote.group(1))
    return _clean_inline(line)


def _strip_outer_braces(cell: str) -> str:
    r"""{$R$, \si{m}} -> $R$, \si{m} (LaTeX grouping around a whole cell)."""
    while cell.startswith("{") and cell.endswith("}"):
        cell = cell[1:-1].strip()
    return cell


def _tabular_to_markdown(body: str) -> str:
    """LaTeX tabular -> markdown pipe rows, so the box renderer can take over."""
    body = re.sub(r"\\(?:top|mid|bottom)rule|\\(?:hline|cline)\s*(?:\{[^{}]*\})?", "", body)
    body = re.sub(r"(?<![A-Za-z])\{([^{}]*)\}", r"\1", body)  # cell grouping braces, not command args
    rows = [row for row in re.split(r"\\\\", body) if row.strip()]
    cells = [[_strip_outer_braces(cell.strip()) for cell in row.split("&")] for row in rows]
    if not cells:
        return ""
    columns = max(len(row) for row in cells)
    lines = ["| " + " | ".join(row + [""] * (columns - len(row))) + " |" for row in cells]
    lines.insert(1, "| " + " | ".join(["---"] * columns) + " |")
    return "\n".join(lines)


def _latex_body_to_markdown(lines: list[str]) -> list[str]:
    """Reduce a pasted LaTeX document to renderable markdown-ish text."""
    text = "\n".join(lines)
    if "\\begin{document}" in text:
        text = text.split("\\begin{document}", 1)[1]
    text = text.split("\\end{document}", 1)[0]
    text = re.sub(
        r"\\begin\{tabular\}(?:\{[^{}]*\})?(.*?)\\end\{tabular\}",
        lambda m: f"\n{_tabular_to_markdown(m.group(1))}\n",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\(?:documentclass|usepackage|begin|end|noindent|medskip|bigskip|smallskip"
        r"|maketitle|centering|label|protect|vspace|hspace)\s*(?:\[[^\]]*\])?\s*(?:\{[^{}]*\})?",
        "",
        text,
    )
    text = re.sub(r"\\section\*?\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:caption|ref|cite|eqref)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"^\s*\[[htbp!]{1,4}\]\s*$", "", text, flags=re.MULTILINE)  # float placement
    text = re.sub(r"\\(?:textbf|textit|emph|mathrm|text)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\SI\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1 \2", text)
    text = re.sub(r"\\si\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:left|right)?\{([^{}]*)\}", r"\1", text)  # grouping braces in cells
    return _join_unbalanced_braces(text)


def _join_unbalanced_braces(text: str) -> list[str]:
    """Models write multi-line fractions/sqrt; glue lines until their braces close."""
    merged: list[str] = []
    buffer = ""
    for line in text.split("\n"):
        buffer = f"{buffer} {line.strip()}".strip() if buffer else line.strip()
        if buffer.count("{") > buffer.count("}"):
            continue
        merged.append(buffer)
        buffer = ""
    if buffer:
        merged.append(buffer)
    return [line.rstrip() for line in merged]


RENDERABLE_FENCE_LANGUAGES = {"markdown", "md", "latex", "tex"}


def _unwrap_renderable_fences(text: str) -> str:
    """DeepSeek often wraps the whole answer in a ```markdown/```latex fence."""
    lines = text.split("\n")
    out: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not (stripped.startswith("```") or stripped.startswith("~~~")):
            out.append(lines[index])
            index += 1
            continue
        language = stripped.lstrip("`~").strip().lower()
        body: list[str] = []
        nested = False
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            fence_here = candidate.startswith("```") or candidate.startswith("~~~")
            if fence_here:
                tag = candidate.lstrip("`~").strip().lower()
                if not nested and not tag:
                    break  # bare fence at wrapper level closes it
                nested = bool(tag)  # a language-tagged fence opens a nested block
            body.append(lines[index])
            index += 1
        index += 1  # consume the closing fence
        if language in {"markdown", "md"}:
            out.extend(body)
        elif language in {"latex", "tex"}:
            out.extend(_latex_body_to_markdown(body))
        else:
            out.append(stripped)
            out.extend(body)
            out.append("```")
    return "\n".join(out)


def _render_markdown(text: str) -> str:
    """Make an answer readable in hosts that print tool results as literal text.

    Markdown tables become box-drawing tables (a raw pipe table wraps and loses
    its columns), heading/bold/link markers are dropped, and math is rendered.
    Fenced code is left exactly as-is — except a ```markdown fence, which is the
    model wrapping its own answer and should be unwrapped and rendered.
    """
    lines = _unwrap_renderable_fences(text).split("\n")
    out: list[str] = []
    index = 0
    in_code = False
    in_math = False
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        is_fence = stripped.startswith("```") or stripped.startswith("~~~")
        if not in_math and is_fence:
            in_code = not in_code
            out.append(raw)
            index += 1
            continue
        if in_code:
            out.append(raw)
            index += 1
            continue
        if in_math and stripped in {"$$", "\\]"}:
            in_math = False
            index += 1
            continue
        if not in_math and stripped in {"$$", "\\["}:
            in_math = True
            index += 1
            continue
        if in_math:
            out.append(_render_math(raw))
            index += 1
            continue
        if stripped.startswith("|"):
            block: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index])
                index += 1
            if len(block) >= 2 and _is_table_separator(block[1]):
                out.extend(_box_table(block))
                continue
            out.extend(block)
            continue
        out.append(_clean_line(raw))
        index += 1
    return "\n".join(out)


def _build_content(question: str, image_path: str | None) -> str | list[dict[str, object]]:
    if not image_path:
        return question
    path = Path(image_path)
    if not path.is_file():
        raise ValueError(f"image_path does not exist: {image_path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
    ]


@server.tool()
async def ask_deepseek(
    question: str,
    ctx: Context,
    new_conversation: bool = False,
    image_path: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT,
) -> str:
    """Ask DeepSeek Web through the local gateway.

    Reasoning and web search are always on. While the model thinks, a compact
    one-line ticker with the newest reasoning slice is reported as progress
    messages (pi replaces its status line in place). Hosts that cannot receive
    progress get a bounded slice of the chain (4k head + 2k tail plus an
    omitted-chars marker) appended to the result, dimmed, after the answer.

    The answer always comes first, with markdown tables redrawn as box-drawing
    tables: hosts print results as literal text and may show only the first
    lines of a collapsed tool block.
    """
    async with _lock:
        if new_conversation:
            _history.clear()
        try:
            model = MODEL
            token = read_api_token(_settings)
            content = _build_content(question, image_path)
        except Exception as exc:
            return f"deepseek-local-server configuration error: {exc}"

        user = {"role": "user", "content": content}
        messages = [*_history, user]

        reasoning_acc: list[str] = []
        answer_acc: list[str] = []
        tool_markup: list[str] = []
        error_text: str | None = None
        live = _can_receive_progress(ctx)
        last_notify = 0.0
        thinking_done = False

        async def _tick(text: str, *, force: bool = False) -> None:
            """Show the newest slice of reasoning as one compact status line."""
            nonlocal last_notify
            if not live or not text:
                return
            now = time.monotonic()
            if not force and now - last_notify < NOTIFY_INTERVAL_SECONDS:
                return
            last_notify = now
            flat = " ".join(text.split())  # hosts append status text into the transcript
            try:
                await ctx.report_progress(
                    progress=float(len(text)), message=f"thinking: …{flat[-NOTIFY_TAIL_CHARS:]}"
                )
            except Exception:
                pass  # notifications are best-effort

        async def _consume(client: httpx.AsyncClient) -> None:
            nonlocal error_text, thinking_done
            async with client.stream(
                "POST",
                f"{_settings.api_base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}", "x-agent-session": _session_id},
                json={"model": model, "messages": messages, "stream": True},
            ) as response:
                if response.is_error:
                    body = (await response.aread()).decode("utf-8", "replace")
                    error_text = f"deepseek-local-server error {response.status_code}: {body}"
                    return
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event.get("error"), dict):
                            error_text = f"deepseek-local-server error: {event['error'].get('message', data)}"
                            return
                        choices = event.get("choices") or [{}]
                        delta = choices[0].get("delta") or {}
                        if delta.get("reasoning_content"):
                            reasoning_acc.append(delta["reasoning_content"])
                            await _tick("".join(reasoning_acc))
                        if delta.get("content"):
                            if not thinking_done:
                                thinking_done = True
                                await _tick("".join(reasoning_acc), force=True)  # thinking finished
                            answer_acc.append(delta["content"])
                        if delta.get("tool_calls"):
                            tool_markup.append(json.dumps(delta["tool_calls"], ensure_ascii=False))

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds + 10) as client:
                task = asyncio.create_task(_consume(client))
                try:
                    await asyncio.wait({task})
                    task.result()
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        except httpx.ConnectError:
            return f"Could not reach deepseek-local-server at {_settings.api_base_url}. Start `deepseek-local-server serve`."
        except httpx.TimeoutException:
            return f"Timed out after {timeout_seconds:.0f}s."

        if error_text:
            return error_text
        answer = "".join(answer_acc) or "".join(tool_markup)
        if answer:
            _history.extend([user, {"role": "assistant", "content": answer}])
        answer = _render_markdown(answer) if answer else "(DeepSeek returned an empty response)"
        reasoning_text = "".join(reasoning_acc)
        if reasoning_text and not live:
            # No progress channel at all: append the chain, but keep the answer first —
            # hosts show only the first lines of a collapsed result.
            block = f"<reasoning>\n{_trim_reasoning(reasoning_text)}\n</reasoning>"
            return f"{answer}\n\n---\n{_dim(block)}"
        return answer


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
