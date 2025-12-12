from typing import Any, Dict, Optional

from diffusers.utils.state_dict_utils import (
    DIFFUSERS_TO_PEFT,
    DIFFUSERS_OLD_TO_PEFT,
)
from src.converters.transformer_converters import TransformerConverter

BASE_TO_PEFT = {
    "lora_down": "lora_A",
    "lora_up": "lora_B",
}

from enum import Enum

class StateDictType(Enum):
    DIFFUSERS = "diffusers"
    DIFFUSERS_OLD = "diffusers_old"
    KOHYA_SS = "kohya_ss"
    PEFT = "peft"
    BASE = "base"

class LoraConverter(TransformerConverter):
    """
    Utility to normalize arbitrary LoRA state dicts into PEFT format.

    This mirrors the behaviour of diffusers' `state_dict_utils` helpers but
    performs the conversion *in-place* on the provided state dict, avoiding the
    creation of a full secondary state dict of the same size.
    """
    def __init__(self):
        super().__init__()
        
    def _get_state_dict_type(self, state_dict:Dict[str, Any]) -> StateDictType:
        """
        Detect the state dict type by checking if any keys match patterns
        characteristic of each format.
        """
        keys = list(state_dict.keys())
        
        # Check for KOHYA_SS patterns (most distinctive)
        kohya_patterns = ["lora_te1.", "lora_te2.", "lora_unet", "dora_scale"]
        if any(any(pattern in key for pattern in kohya_patterns) for key in keys):
            return StateDictType.KOHYA_SS
        
        base_patterns = ["lora_down", "lora_up"]
        if any(any(pattern in key for pattern in base_patterns) for key in keys):
            return StateDictType.BASE
        
        # Check for DIFFUSERS_OLD patterns
        diffusers_old_patterns = [".to_q_lora", ".to_k_lora", ".to_v_lora", ".to_out_lora"]
        if any(any(pattern in key for pattern in diffusers_old_patterns) for key in keys):
            return StateDictType.DIFFUSERS_OLD
        
        # Check for DIFFUSERS patterns
        diffusers_patterns = [".lora_linear_layer.up", ".lora_linear_layer.down"]
        if any(any(pattern in key for pattern in diffusers_patterns) for key in keys):
            return StateDictType.DIFFUSERS
        
        # Check for PEFT patterns (default fallback)
        peft_patterns = [".lora_A", ".lora_B"]
        if any(any(pattern in key for pattern in peft_patterns) for key in keys):
            return StateDictType.PEFT
        
        # Default to PEFT if no patterns match
        return StateDictType.PEFT
    
    def convert(self, state_dict: Dict[str, Any]):
        state_dict_type = self._get_state_dict_type(state_dict)
        if state_dict_type == StateDictType.KOHYA_SS:
            convert_kohya_to_peft_state_dict(state_dict)
        elif state_dict_type == StateDictType.DIFFUSERS_OLD:
            self.rename_dict = DIFFUSERS_OLD_TO_PEFT
            super().convert(state_dict)
        elif state_dict_type == StateDictType.DIFFUSERS:
            self.rename_dict = DIFFUSERS_TO_PEFT
            super().convert(state_dict)
        elif state_dict_type == StateDictType.BASE:
            self.rename_dict = BASE_TO_PEFT
            super().convert(state_dict)
        elif state_dict_type == StateDictType.PEFT:
            pass
        return state_dict


def convert_kohya_to_peft_state_dict(
    kohya_state_dict: Dict[str, Any],
    adapter_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Best-effort reverse of `convert_state_dict_to_kohya` from diffusers:

        Kohya-format LoRA -> PEFT-format LoRA.

    Behaviour:
    - Drops Kohya `.alpha` tensors (they don't exist in PEFT weights).
    - Restores `lora_te1.` / `lora_te2.` / `lora_unet` headers back to
      `text_encoder.` / `text_encoder_2.` / `unet`.
    - Restores `dora_scale` back to `lora_magnitude_vector`.
    - Re-introduces dots that were flattened into underscores in the Kohya
      format, following the same “keep the last two dots” convention.
    - Maps `lora_down` / `lora_up` back to `lora_A` / `lora_B`.
    - Optionally injects an `adapter_name` segment before the final
      `.weight` / `.bias`.
    """
    items = list(kohya_state_dict.items())
    kohya_state_dict.clear()

    for key, value in items:
        # Skip Kohya alpha scalars – PEFT will infer / store scaling separately.
        if key.endswith(".alpha"):
            continue

        k = key

        # Undo header conversions.
        if k.startswith("lora_te2."):
            k = k.replace("lora_te2.", "text_encoder_2.", 1)
        elif k.startswith("lora_te1."):
            k = k.replace("lora_te1.", "text_encoder.", 1)
        elif k.startswith("lora_unet"):
            k = k.replace("lora_unet", "unet", 1)

        # Undo DoRA naming.
        if "dora_scale" in k:
            k = k.replace("dora_scale", "lora_magnitude_vector")

        # Restore dots in the prefix part, mirroring the logic from
        # `convert_state_dict_to_kohya`, which replaces all but the last two
        # dots with underscores.
        last_dot = k.rfind(".")
        if last_dot != -1:
            second_last_dot = k.rfind(".", 0, last_dot)
            if second_last_dot != -1:
                prefix = k[:second_last_dot]
                tail = k[second_last_dot:]  # includes the dot

                prefix = prefix.replace("_", ".")
                k = prefix + tail

        # Undo lora_down / lora_up mapping.
        k = k.replace(".lora_down", ".lora_A")
        k = k.replace(".lora_up", ".lora_B")

        # Optionally reinsert adapter name before the final weight/bias token.
        if adapter_name and (k.endswith(".weight") or k.endswith(".bias")):
            base, suffix = k.rsplit(".", 1)
            k = f"{base}.{adapter_name}.{suffix}"

        kohya_state_dict[k] = value

    return kohya_state_dict


