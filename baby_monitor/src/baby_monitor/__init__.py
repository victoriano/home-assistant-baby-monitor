"""Baby Monitor for Home Assistant backend."""

from typing import Any

__all__ = ["create_app"]
__version__ = "0.1.0"


def create_app(**kwargs: Any) -> Any:
    """Lazy factory that avoids creating /data merely by importing a model."""

    from .main import create_app as factory

    return factory(**kwargs)
