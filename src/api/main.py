from fastapi import FastAPI
from .ws import router as ws_router
from .manifest import router as manifest_router
from .config import router as config_router
from .preprocessor import router as preprocessor_router
from fastapi.middleware.cors import CORSMiddleware
from .ray_app import get_ray_app, shutdown_ray
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Ray
    get_ray_app()
    
    # Initialize the Ray websocket bridge
    from .ws_manager import get_ray_ws_bridge
    get_ray_ws_bridge()
    
    # Initialize preprocessor download tracking
    from .preprocessor_registry import initialize_download_tracking
    initialize_download_tracking()
    
    # Start background task for polling Ray updates
    from .preprocessor import poll_ray_updates
    poll_task = asyncio.create_task(poll_ray_updates())
    
    yield
    
    # Shutdown: Cancel polling task and close Ray
    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass
    shutdown_ray()

app = FastAPI(name="Apex Engine", lifespan=lifespan)
app.include_router(ws_router)
app.include_router(manifest_router)
app.include_router(config_router)
app.include_router(preprocessor_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    
@app.get("/health")
def read_root():
    return {"status": "ok"}
