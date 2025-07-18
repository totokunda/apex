import torch
from typing import Dict, Any, Callable, List, Union, Optional
from PIL import Image
import numpy as np

from .base import SkyReelsBaseEngine


class SkyReelsI2VEngine(SkyReelsBaseEngine):
    """SkyReels Image-to-Video Engine Implementation"""
    
    def run(self, **kwargs):
        """Image-to-video generation for SkyReels model"""
        # Override with fps=24 as per the original implementation  
        kwargs["fps"] = kwargs.get("fps", 24)
        return self.main_engine.i2v_run(**kwargs) 