"""
LoRA utilities.

Note: Keep this package import lightweight so that submodules (e.g. `key_remap`)
can be imported without pulling heavy optional dependencies.
"""

try:
    from .manager import LoraManager, LoraItem

    __all__ = ["LoraManager", "LoraItem"]
except Exception:
    # Allow importing lightweight submodules even if optional deps required by
    # `manager` aren't installed in the current environment.
    __all__ = []
