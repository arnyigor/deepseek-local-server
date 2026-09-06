import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from deepseek_local_server.browser.worker import BrowserResult
from deepseek_local_server.config import Settings
from deepseek_local_server.errors import BrowserProtocolError
from deepseek_local_server.openai.schemas import ChatCompletionRequest
from deepseek_local_server.service import CompletionService


def test_request_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="deepseek-web", messages=[{"role": "user", "content": "hi"}], mode="unknown")


@pytest.mark.parametrize("mode,label", [("instant", "Instant"), ("expert", "Expert")])
def test_mode_reaches_browser_and_only_expert_enables_deepthink(tmp_path, mode, label):
    service = CompletionService(Settings(home=tmp_path))
    worker = service.worker
    page = Mock()
    page.is_closed.return_value = False
    option = AsyncMock()
    option.count.return_value = 1
    option.is_visible.return_value = True
    page.get_by_text.return_value.first = option
    worker._ensure_deepthink = AsyncMock()
    worker._find_composer = AsyncMock()
    worker._message_snapshot = AsyncMock(return_value=[])
    worker._count = AsyncMock(return_value=0)
    worker._fill_composer = AsyncMock()
    worker._submit_prompt = AsyncMock(return_value="send_button")
    worker._wait_for_response = AsyncMock(return_value=BrowserResult("answer", "https://chat.deepseek.com/test"))
    # Use an existing chat to also cover switching away from its previous mode.
    initial = ChatCompletionRequest(model="deepseek-web", messages=[{"role": "user", "content": "first"}])
    service._session_page = page
    service._session_messages = initial.messages
    request = initial.model_copy(update={
        "mode": mode,
        "messages": [*initial.messages, initial.messages[0].model_copy(update={"content": "continue"})],
    })

    result = asyncio.run(service.complete(request))

    assert result.parsed.content == "answer"
    page.get_by_text.assert_called_once_with(label, exact=True)
    option.click.assert_awaited_once()
    if mode == "expert":
        worker._ensure_deepthink.assert_awaited_once_with(page)
    else:
        worker._ensure_deepthink.assert_not_awaited()
    assert service._session_page is page


def test_missing_instant_does_not_silently_fall_back_to_expert(tmp_path):
    worker = CompletionService(Settings(home=tmp_path)).worker
    page = Mock()
    option = AsyncMock()
    option.count.return_value = 0
    page.get_by_text.return_value.first = option

    with pytest.raises(BrowserProtocolError, match="Instant"):
        asyncio.run(worker._ensure_model(page, "instant"))

    option.click.assert_not_awaited()
