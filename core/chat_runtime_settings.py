from typing import Any, Dict


def _int_setting(settings: Dict[str, Any], key: str, default: int, min_value: int, max_value: int) -> int:
    value = settings.get(key, default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def resolve_chat_runtime_settings(settings: Dict[str, Any]) -> Dict[str, int]:
    return {
        "chat_topk": _int_setting(settings, "CHAT_TOPK", default=3, min_value=1, max_value=20),
        "compare_topk": _int_setting(settings, "COMPARE_TOPK", default=12, min_value=1, max_value=50),
        "chat_max_tokens": _int_setting(settings, "CHAT_MAX_TOKENS", default=1200, min_value=64, max_value=4096),
        "chat_history_rounds": _int_setting(settings, "CHAT_HISTORY_ROUNDS", default=3, min_value=1, max_value=20),
    }
