# 兼容层：保留旧 import 路径 evaluator.usage_logger，统一转发到顶层 usage_logger。
# 注意：本文件本身不能在 evaluator/__init__.py 加载时被解析，
# 我们仅在用户显式 `from evaluator.usage_logger import xxx` 时才进入。
from usage_logger import (  # noqa: F401
    log_usage,
    log_event,
    extract_usage,
    format_usage_line,
    usage_to_langfuse,
    USAGE_LOG_ENABLED,
    USAGE_LOG_FILE,
)
