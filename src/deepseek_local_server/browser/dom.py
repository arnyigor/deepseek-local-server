from __future__ import annotations

import re

# Confirmed from probe dumps (probe_send2.log / probe-after-send-*.html) — see plan.md.
ASSISTANT_SELECTORS = [
    ".ds-markdown.ds-assistant-message-main-content",
]

# Baseline message-count check only (see plan.md note on the hashed 'fbb737a4' user class).
USER_MESSAGE_COUNT_SELECTOR = ".ds-message"

COMPOSER_SELECTORS = [
    'textarea[placeholder="Message DeepSeek"]',
    "textarea:not([disabled])",
]

# Not a real <button> -- a styled <div>. Enter/Control+Enter is the proven fallback (see
# plan.md); this selector is tried first as a best-effort, same order as qwen.
SEND_SELECTORS = [
    "div.ds-button--primary",
]

NEW_CHAT_SELECTORS = [
    'button[aria-label*="New chat" i]',
    'button[title*="New chat" i]',
    'button:has-text("New chat")',
]

# Unknown -- no mid-generation dump was captured. Left empty on purpose; _wait_for_response
# degrades to pure text-stability polling, which is already proven to work (see plan.md).
STOP_SELECTORS: list[str] = []

# The "expert" mode toggle. State lives in aria-pressed on this same element.
DEEPTHINK_TOGGLE_SELECTOR = 'div.ds-toggle-button:has(span:text-is("DeepThink"))'

SNAPSHOT_MESSAGES_JS = r"""
(args) => {
  const { selectors } = args;
  const visible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.visibility !== 'hidden' && style.display !== 'none' &&
      rect.width > 0 && rect.height > 0;
  };

  const elements = [];
  const seen = new Set();
  for (const selector of selectors) {
    for (const el of document.querySelectorAll(selector)) {
      if (seen.has(el) || !visible(el)) continue;
      seen.add(el);
      const text = (el.innerText || el.textContent || '').trim();
      if (!text) continue;
      const rect = el.getBoundingClientRect();
      elements.push({ text, top: rect.top + window.scrollY, selector });
    }
  }

  elements.sort((a, b) => a.top - b.top);
  const result = [];
  for (const item of elements) {
    const duplicate = result.some(
      (existing) => existing.text === item.text && Math.abs(existing.top - item.top) < 6
    );
    if (!duplicate) result.push(item);
  }
  return result;
}
"""

PAGE_ERROR_JS = r"""
() => {
  // Scan only toast/notification-style elements, never the whole page body: the conversation
  // itself (the user's own submitted prompt, echoed back in the transcript) routinely contains
  // phrases like "network error" or "something went wrong" as ordinary text, which would cause
  // false-positive failures when scanning document.body.innerText directly.
  const nodes = document.querySelectorAll(
    '[class*="notification"], [class*="toast"], [role="alert"], [role="status"]'
  );
  const text = Array.from(nodes)
    .map((el) => el.innerText || '')
    .join(' ')
    .toLowerCase();
  const patterns = [
    ['auth', ['login expired', 'sign in to continue', 'log in to continue']],
    ['deepseek', ['something went wrong', 'please try again later', 'network error', 'server is busy']],
  ];
  for (const [kind, variants] of patterns) {
    if (variants.some((value) => text.includes(value))) return kind;
  }
  return null;
}
"""

_JUNK_TEXT_PATTERN = re.compile(r"^\+\d+$")


def is_junk_snapshot_text(text: str) -> bool:
    stripped = text.strip()
    return bool(_JUNK_TEXT_PATTERN.match(stripped))


CLEAN_PREFIXES = (
    "Thinking completed",
    "Thinking...",
    "深度思考完成",
)


def clean_assistant_text(text: str) -> str:
    result = text.strip()
    changed = True
    while changed:
        changed = False
        for prefix in CLEAN_PREFIXES:
            if result.startswith(prefix):
                result = result[len(prefix):].lstrip("\n \t:-")
                changed = True
    result = result.strip()
    if is_junk_snapshot_text(result):
        return ""
    return result
