"""Scene encoder construction."""

from .scene_encoder import SceneEncoder

__all__ = ["SceneEncoder", "build_context_encoder"]

_ENCODERS = {
    "SceneEncoder": SceneEncoder,
}


def build_context_encoder(config):
    return _ENCODERS[config.NAME](config=config)
