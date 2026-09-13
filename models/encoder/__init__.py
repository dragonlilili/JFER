# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


from .scene_encoder import SceneEncoder

__all__ = {
    'SceneEncoder': SceneEncoder,
}


def build_context_encoder(config):
    model = __all__[config.NAME](
        config=config
    )

    return model
