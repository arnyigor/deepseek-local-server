class DeepSeekLocalServerError(RuntimeError):
    """Base error for expected local-server failures."""


class AuthenticationRequiredError(DeepSeekLocalServerError):
    pass


class BrowserProtocolError(DeepSeekLocalServerError):
    pass


class DeepSeekPageError(DeepSeekLocalServerError):
    pass


class UnsupportedContentError(DeepSeekLocalServerError):
    pass


class ToolProtocolError(DeepSeekLocalServerError):
    pass
