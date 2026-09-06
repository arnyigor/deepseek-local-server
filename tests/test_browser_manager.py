from deepseek_local_server.browser.manager import BrowserManager


class FakePage:
    def __init__(self, url: str, closed: bool = False) -> None:
        self.url = url
        self._closed = closed

    def is_closed(self) -> bool:
        return self._closed


def test_live_pages_filters_closed_pages() -> None:
    open_page = FakePage("https://chat.deepseek.com/")
    closed_page = FakePage("https://example.test/", closed=True)

    result = BrowserManager._live_pages([closed_page, open_page])  # type: ignore[arg-type]

    assert result == [open_page]
