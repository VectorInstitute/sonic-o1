import os
from .siglip_encoder import SigLipVisionTower

# from .eva_clip.eva_clip_encoder import EvaClipVisionTower
# from .dev_eva_clip.eva_vit import EvaViTWrapper


def build_vision_tower(vision_tower_cfg, **kwargs):
    return SigLipVisionTower(vision_tower_cfg=vision_tower_cfg, **kwargs)

