import torch
from typing import List

from .base import SkyReelsBaseEngine


class SkyReelsT2VEngine(SkyReelsBaseEngine):
    """SkyReels Text-to-Video Engine Implementation"""

    def run(self, **kwargs):
        """Text-to-video generation for SkyReels model"""
        # Override with fps=24 as per the original implementation
        kwargs["fps"] = kwargs.get("fps", 24)
        return self.main_engine.t2v_run(**kwargs)
