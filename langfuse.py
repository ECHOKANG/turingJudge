"""No-op Langfuse shim.

The app does not require Langfuse at runtime. This local module shadows the
external package and keeps existing tracing calls from affecting evaluation.
"""


class _NoopClient:
    def update_current_trace(self, *args, **kwargs):
        return None

    def update_current_observation(self, *args, **kwargs):
        return None

    def flush(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop


class Langfuse(_NoopClient):
    pass


_client = _NoopClient()


def get_client(*args, **kwargs):
    return _client


def observe(_func=None, *args, **kwargs):
    if callable(_func):
        return _func

    def _decorator(func):
        return func

    return _decorator
