class DeepSeekLocalError(RuntimeError):
    """Base error for the local gateway."""


class AuthenticationRequiredError(DeepSeekLocalError):
    pass


class DirectBackendError(DeepSeekLocalError):
    pass


class DirectProtocolError(DirectBackendError):
    pass


class DirectRateLimitError(DirectBackendError):
    pass


class BrowserProtocolError(DeepSeekLocalError):
    pass


class ToolProtocolError(DeepSeekLocalError):
    pass
