import os
import json
import ray
from .settings import settings
from loguru import logger

# Initialize Ray if not already initialized
if not ray.is_initialized():
    # Prepare spilling config directory
    spill_dir = settings.results_dir / "ray_spill"
    spill_dir.mkdir(parents=True, exist_ok=True)

    # Try to initialize with dashboard, fall back without if packages missing
    try:
        ray.init(
            ignore_reinit_error=True,
            num_cpus=os.cpu_count(),
            dashboard_host="0.0.0.0",
            dashboard_port=settings.ray_dashboard_port,
            include_dashboard=True,
            _system_config={
                "automatic_object_spilling_enabled": True,
                "object_spilling_config": json.dumps(
                    {"type": "filesystem", "params": {"directory_path": str(spill_dir)}}
                ),
            },
        )
        logger.info(f"Ray initialized with {ray.available_resources()}")
        logger.info(
            f"Ray dashboard available at http://localhost:{settings.ray_dashboard_port}"
        )
    except Exception as e:
        if "Cannot include dashboard" in str(e):
            logger.warning(
                "Ray dashboard packages not installed, starting without dashboard"
            )
            logger.info(
                "To enable dashboard: pip install 'ray[default,dashboard]' aiohttp aiohttp-cors grpcio"
            )
            ray.init(
                ignore_reinit_error=True,
                num_cpus=os.cpu_count(),
                include_dashboard=False,
                _system_config={
                    "automatic_object_spilling_enabled": True,
                    "object_spilling_config": json.dumps(
                        {
                            "type": "filesystem",
                            "params": {"directory_path": str(spill_dir)},
                        }
                    ),
                },
            )
            logger.info(f"Ray initialized with {ray.available_resources()}")
        else:
            raise


def get_ray_app():
    """Get the Ray runtime"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    return ray


def shutdown_ray():
    """Shutdown Ray runtime"""
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown complete")
