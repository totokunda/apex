from typing import Dict, Type, Any, Optional, List, Literal
from enum import Enum
import torch
from src.ui.nodes import UINode

class EngineType(Enum):
    """Supported engine types"""
    WAN = "wan"
    HUNYUAN = "hunyuan"
    LTX = "ltx" 
    COGVIDEO = "cogvideo"
    MAGI = "magi"
    STEPVIDEO = "stepvideo"
    MOCHI = "mochi"
    SKYREELS = "skyreels"


class EngineRegistry:
    """Central registry for all engine implementations"""
    
    def __init__(self):
        self._engines: Dict[str, Type] = {}
        self._register_engines()
    
    def _register_engines(self):
        """Register all available engines"""
        
        # Register WAN engine
        try:
            from src.engine.wan import WanEngine
            self._engines[EngineType.WAN.value] = WanEngine
        except ImportError as e:
            print(f"Warning: Could not import WAN engine: {e}")
        
        # Register Hunyuan engine
        try:
            from src.engine.hunyuan import HunyuanEngine
            self._engines[EngineType.HUNYUAN.value] = HunyuanEngine
        except ImportError as e:
            print(f"Warning: Could not import Hunyuan engine: {e}")
        
        # Register LTX engine
        try:
            from src.engine.ltx import LTXEngine
            self._engines[EngineType.LTX.value] = LTXEngine
        except ImportError as e:
            print(f"Warning: Could not import LTX engine: {e}")
        
        # Register CogVideo engine
        try:
            from src.engine.cogvideo import CogVideoEngine
            self._engines[EngineType.COGVIDEO.value] = CogVideoEngine
        except ImportError as e:
            print(f"Warning: Could not import CogVideo engine: {e}")
        
        # Register Magi engine
        try:
            from src.engine.magi import MagiEngine
            self._engines[EngineType.MAGI.value] = MagiEngine
        except ImportError as e:
            print(f"Warning: Could not import Magi engine: {e}")
        
        # Register StepVideo engine
        try:
            from src.engine.stepvideo import StepVideoEngine
            self._engines[EngineType.STEPVIDEO.value] = StepVideoEngine
        except ImportError as e:
            print(f"Warning: Could not import StepVideo engine: {e}")
        
        # Register Mochi engine
        try:
            from src.engine.mochi import MochiEngine
            self._engines[EngineType.MOCHI.value] = MochiEngine
        except ImportError as e:
            print(f"Warning: Could not import Mochi engine: {e}")
        
        # Register SkyReels engine
        try:
            from src.engine.skyreels import SkyReelsEngine
            self._engines[EngineType.SKYREELS.value] = SkyReelsEngine
        except ImportError as e:
            print(f"Warning: Could not import SkyReels engine: {e}")

    def get_engine_class(self, engine_type: str) -> Optional[Type]:
        """Get engine class by type"""
        return self._engines.get(engine_type)
    
    def list_engines(self) -> List[str]:
        """List all available engine types"""
        return list(self._engines.keys())
    
    def create_engine(
        self, 
        engine_type: str, 
        yaml_path: str, 
        model_type: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create an engine instance"""
        engine_class = self.get_engine_class(engine_type)
        if engine_class is None:
            raise ValueError(f"Unknown engine type: {engine_type}")
        
        # For engines that support model_type
        if model_type is not None:
            kwargs['model_type'] = self._get_model_type_enum(engine_type, model_type)
        
        return engine_class(yaml_path=yaml_path, **kwargs)
    
    def _get_model_type_enum(self, engine_type: str, model_type: str):
        """Get the appropriate ModelType enum for an engine"""
        if engine_type == EngineType.WAN.value:
            from src.engine.wan import ModelType
            return ModelType(model_type)
        elif engine_type == EngineType.HUNYUAN.value:
            from src.engine.hunyuan import ModelType
            return ModelType(model_type)
        elif engine_type == EngineType.LTX.value:
            from src.engine.ltx import ModelType
            return ModelType(model_type)
        elif engine_type == EngineType.COGVIDEO.value:
            from src.engine.cogvideo import ModelType
            return ModelType(model_type)
        elif engine_type == EngineType.MAGI.value:
            from src.engine.magi import ModelType
            return ModelType(model_type)
        elif engine_type == EngineType.STEPVIDEO.value:
            from src.engine.stepvideo import ModelType
            return ModelType(model_type)
        elif engine_type == EngineType.SKYREELS.value:
            from src.engine.skyreels import ModelType
            return ModelType(model_type)
        else:
            # Fall back to string for engines without ModelType enum
            return model_type


class UniversalEngine:
    """Universal engine interface that can run any registered engine"""
    
    def __init__(self, engine_type: str, yaml_path: str, model_type: Optional[str] = None, **kwargs):
        self.registry = EngineRegistry()
        self.engine = self.registry.create_engine(
            engine_type=engine_type,
            yaml_path=yaml_path,
            model_type=model_type,
            **kwargs
        )
        self.engine_type = engine_type
        self.model_type = model_type
    
    @torch.inference_mode()
    def run(self, input_nodes: List[UINode] | None = None, **kwargs):
        """Run the engine with given parameters"""
        return self.engine.run(input_nodes=input_nodes, **kwargs)
    
    def load_component_by_type(self, component_type: str):
        """Load a component by type"""
        return self.engine.load_component_by_type(component_type)
    
    def set_attention_type(self, attention_type: str):
        """Set attention type if supported"""
        if hasattr(self.engine, 'set_attention_type'):
            return self.engine.set_attention_type(attention_type)
    
    def to_device(self, device: torch.device):
        """Move engine to device if supported"""
        if hasattr(self.engine, 'to'):
            return self.engine.to(device)
    
    def __getattr__(self, name):
        """Delegate any missing attributes to the underlying engine"""
        return getattr(self.engine, name)
    
    def __str__(self):
        return f"UniversalEngine(engine_type={self.engine_type}, model_type={self.model_type})"
    
    def __repr__(self):
        return self.__str__()


# Global registry instance
_global_registry = EngineRegistry()


def get_engine_registry() -> EngineRegistry:
    """Get the global engine registry"""
    return _global_registry


def create_engine(engine_type: Literal["wan", "hunyuan", "ltx", "cogvideo", "magi", "stepvideo", "mochi", "skyreels"], yaml_path: str, model_type: str | None = None, **kwargs) -> UniversalEngine:
    """Convenience function to create an engine"""
    return _global_registry.create_engine(
        engine_type=engine_type,
        yaml_path=yaml_path, 
        model_type=model_type,
        **kwargs
    )


def list_available_engines() -> List[str]:
    """List all available engine types"""
    return _global_registry.list_engines() 