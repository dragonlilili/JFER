"""Public construction interface for the JFER motion decoder."""

from __future__ import annotations

from typing import Any


def build_jfer_decoder(in_channels: int, config: Any) -> Any:
    """Build the JFER motion decoder."""
    from .modules.jfer_decoder import JFERDecoder

    return JFERDecoder(in_channels=in_channels, config=config)


__all__ = ["build_jfer_decoder"]
