import torch
import math
import numpy as np
from src.utils.type_utils import EnumType
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from tqdm import tqdm
import gc


class MagiDenoiseType(EnumType):
    BASE = "base"


class MagiDenoise:
    def __init__(
        self, denoise_type: MagiDenoiseType = MagiDenoiseType.T2V, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.denoise_type = denoise_type

    def denoise(self, *args, **kwargs) -> torch.Tensor:
        """Unified denoising method that handles chunk-based generation"""
        # Store MAGI runtime config if provided
        if self.denoise_type == MagiDenoiseType.BASE:
            return self.base_denoise(*args, **kwargs)
        else:
            raise ValueError(f"Invalid denoise type: {self.denoise_type}")
    
    def base_denoise(self, *args, **kwargs) -> torch.Tensor:
        pass