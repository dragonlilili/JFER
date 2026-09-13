"""Public construction interface for the JFER motion decoder."""

from __future__ import annotations

from typing import Any


def build_jfer_decoder(in_channels: int, config: Any) -> Any:
    """Build JFER after the extension has been installed into MTR."""
    from mtr.models.motion_decoder.jfer_decoder import JFERMTRDecoder

    return JFERMTRDecoder(in_channels=in_channels, config=config)


__all__ = ["build_jfer_decoder"]
