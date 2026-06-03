from langfuse import get_client, observe, Langfuse

_client = None


def get_langfuse():
    global _client
    if _client is None:
        _client = get_client()
    return _client


def flush_langfuse():
    if _client is not None:
        _client.flush()


def _best_effort_client_call(method_name, *args, **kwargs):
    try:
        client = get_langfuse()
        method = getattr(client, method_name, None)
        if callable(method):
            return method(*args, **kwargs)
    except Exception:
        pass
    return None


def _install_langfuse_compat():
    if not hasattr(Langfuse, "update_current_trace"):
        def update_current_trace(self, *args, **kwargs):
            return _best_effort_client_call("update_current_trace", *args, **kwargs)
        Langfuse.update_current_trace = update_current_trace

    if not hasattr(Langfuse, "update_current_observation"):
        def update_current_observation(self, *args, **kwargs):
            return _best_effort_client_call("update_current_observation", *args, **kwargs)
        Langfuse.update_current_observation = update_current_observation


_install_langfuse_compat()
