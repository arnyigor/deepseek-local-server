from deepseek_local_server.browser.dom import (
    DEEPTHINK_TOGGLE_SELECTOR,
    clean_assistant_text,
    is_junk_snapshot_text,
)


def test_junk_source_count_badge_is_filtered() -> None:
    assert is_junk_snapshot_text("+3") is True
    assert is_junk_snapshot_text("Actual answer text") is False


def test_clean_assistant_text_strips_thinking_prefix() -> None:
    assert clean_assistant_text("Thinking completed\nThe answer is 42.") == "The answer is 42."


def test_deepthink_toggle_selector_is_non_empty_and_targets_deepthink() -> None:
    assert DEEPTHINK_TOGGLE_SELECTOR
    assert "DeepThink" in DEEPTHINK_TOGGLE_SELECTOR
    assert "ds-toggle-button" in DEEPTHINK_TOGGLE_SELECTOR
