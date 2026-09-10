from pathlib import Path

import pytest

from deepseek_local_server.config import Settings
from deepseek_local_server.direct.stream import StreamPiece
from deepseek_local_server.openai.schemas import ChatCompletionRequest
from deepseek_local_server.service import CompletionService


class FakeDirect:
    def __init__(self):
        self.prompts = []
        self.fresh_prompts = []
    async def stream(self, prompt, spec, remote, *, fresh_prompt=None, ref_file_ids=None):
        self.prompts.append(prompt)
        self.fresh_prompts.append(fresh_prompt)
        assert fresh_prompt is not None
        remote.id = remote.id or "s1"
        remote.parent_message_id = 101
        remote.message_count += 1
        yield StreamPiece("reasoning", "r")
        yield StreamPiece("content", "answer")


@pytest.mark.asyncio
async def test_service_reuses_only_delta_for_extended_history(tmp_path: Path):
    settings = Settings(home=tmp_path)
    direct = FakeDirect()
    service = CompletionService(settings, direct=direct)

    first = ChatCompletionRequest(model="deepseek-v4-pro", messages=[{"role":"user","content":"one"}])
    result = await service.complete(first, "agent")
    assert result.parsed.content == "answer"
    assert result.reasoning == "r"

    second = ChatCompletionRequest(model="deepseek-v4-pro", messages=[
        {"role":"user","content":"one"},
        {"role":"assistant","content":"answer"},
        {"role":"user","content":"two"},
    ])
    await service.complete(second, "agent")
    assert "two" in direct.prompts[1]
    assert "one" not in direct.prompts[1]
    assert "one" in direct.fresh_prompts[1]
    assert "two" in direct.fresh_prompts[1]
