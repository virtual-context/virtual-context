"""Proxy package — lazy imports to avoid pulling in FastAPI/uvicorn for non-proxy usage."""


def __getattr__(name: str):
    if name in ("create_app", "ProxyState", "SessionRegistry"):
        from .server import create_app, ProxyState, SessionRegistry
        return {"create_app": create_app, "ProxyState": ProxyState, "SessionRegistry": SessionRegistry}[name]
    if name in ("PayloadFormat", "detect_format", "get_format"):
        from .formats import PayloadFormat, detect_format, get_format
        return {"PayloadFormat": PayloadFormat, "detect_format": detect_format, "get_format": get_format}[name]
    if name == "ProxyMetrics":
        from .metrics import ProxyMetrics
        return ProxyMetrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_app",
    "ProxyState",
    "SessionRegistry",
    "ProxyMetrics",
    "PayloadFormat",
    "detect_format",
    "get_format",
]
