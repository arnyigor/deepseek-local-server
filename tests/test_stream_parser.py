from deepseek_local_server.direct.stream import DeepSeekPatchParser


def test_full_snapshot_and_incremental_fragment():
    p = DeepSeekPatchParser()
    pieces = p.feed({"v": {"response": {"message_id": "m1", "fragments": [
        {"type": "THINK", "content": "why"},
        {"type": "RESPONSE", "content": "hel"},
    ]}}})
    assert [(x.kind, x.text) for x in pieces] == [("reasoning", "why"), ("content", "hel")]
    pieces = p.feed({"p": "response/fragments/-1/content", "v": "lo"})
    assert [(x.kind, x.text) for x in pieces] == [("content", "lo")]
    assert p.content == "hello"
    assert p.reasoning == "why"
    assert p.message_id == "m1"


def test_content_fallback_path():
    p = DeepSeekPatchParser()
    assert p.feed({"p": "response/content", "v": "abc"})[0].text == "abc"
    assert p.feed({"p": "response/content", "v": "def"})[0].text == "def"
    assert p.content == "abcdef"


def test_bare_value_deltas_continue_the_last_path():
    # DeepSeek sends "p" only on the first delta of a run; later deltas in the same
    # run arrive as bare {"v": ...} and implicitly continue that path.
    p = DeepSeekPatchParser()
    p.feed({"p": "response", "o": "BATCH", "v": [
        {"p": "fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "The"}]},
    ]})
    p.feed({"p": "response/fragments/-1/content", "o": "APPEND", "v": " first"})
    p.feed({"v": " iPhone"})
    p.feed({"v": " shipped"})
    assert p.content == "The first iPhone shipped"


def test_search_fragment_is_not_answer_text():
    # SEARCH fragments are a "searching..." status marker with content=null; they must
    # not leak a literal "None" into the answer.
    p = DeepSeekPatchParser()
    p.feed({"p": "response/fragments", "v": [{"type": "SEARCH", "status": "WIP", "content": None}]})
    pieces = p.feed({"p": "response", "o": "BATCH", "v": [
        {"p": "fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "Yes"}]},
    ]})
    assert [(x.kind, x.text) for x in pieces] == [("content", "Yes")]
    assert p.content == "Yes"
