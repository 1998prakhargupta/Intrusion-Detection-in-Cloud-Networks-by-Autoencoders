"""API package for the NIDS system."""

try:
    from .main import create_app, start_server
    __all__ = ["create_app", "start_server"]
except ImportError:
    # FastAPI not available
    __all__ = []
