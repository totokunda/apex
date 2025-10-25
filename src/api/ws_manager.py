"""
Websocket manager for handling job status updates
"""
from typing import Dict, Set, Optional
from fastapi.websockets import WebSocket
import ray
from collections import defaultdict

class WebSocketManager:
    """Manages websocket connections for job updates"""
    
    def __init__(self):
        # Map job_id to set of connected websockets
        self.connections: Dict[str, Set[WebSocket]] = {}
        # Store latest updates for each job (for clients connecting late)
        self.latest_updates: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Register a websocket connection for a job"""
        await websocket.accept()
        if job_id not in self.connections:
            self.connections[job_id] = set()
        self.connections[job_id].add(websocket)
        
        # Send latest update if available
        if job_id in self.latest_updates:
            try:
                await websocket.send_json(self.latest_updates[job_id])
            except Exception:
                pass
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Unregister a websocket connection"""
        if job_id in self.connections:
            self.connections[job_id].discard(websocket)
            if not self.connections[job_id]:
                del self.connections[job_id]
    
    async def send_update(self, job_id: str, data: dict):
        """Send update to all websockets listening to a job"""
        # Store latest update
        self.latest_updates[job_id] = data
        
        if job_id in self.connections:
            disconnected = set()
            for websocket in self.connections[job_id]:
                try:
                    await websocket.send_json(data)
                except Exception:
                    disconnected.add(websocket)
            
            # Clean up disconnected websockets
            for websocket in disconnected:
                self.disconnect(websocket, job_id)
    
    def get_latest_update(self, job_id: str) -> Optional[dict]:
        """Get the latest update for a job"""
        return self.latest_updates.get(job_id)


# Global websocket manager instance
websocket_manager = WebSocketManager()


@ray.remote
class RayWebSocketBridge:
    """Ray actor that bridges Ray workers to the websocket manager"""
    
    def __init__(self):
        self.updates: Dict[str, list] = defaultdict(list)
        print("RayWebSocketBridge initialized")
    
    def send_update(self, job_id: str, progress: float, message: str, metadata: Optional[Dict] = None):
        """Store update to be pulled by main process"""
        metadata = metadata or {}
        status = metadata.pop("status", "processing")
        
        update = {
            "progress": progress,
            "message": message,
            "status": status,
            "metadata": metadata
        }
        
        self.updates[job_id].append(update)
        return True
    
    def get_updates(self, job_id: str) -> list:
        """Get all pending updates for a job"""
        updates = self.updates.get(job_id, [])
        self.updates[job_id] = []  # Clear after retrieving
        return updates
    
    def get_all_job_ids(self) -> list:
        """Get all job IDs that have updates"""
        return list(self.updates.keys())
    
    def has_updates(self, job_id: str) -> bool:
        """Check if there are pending updates"""
        return len(self.updates.get(job_id, [])) > 0


# Global Ray actor for websocket bridge
_ray_ws_bridge = None

def get_ray_ws_bridge():
    """Get or create the Ray websocket bridge actor"""
    global _ray_ws_bridge
    if _ray_ws_bridge is None:
        import ray
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before creating websocket bridge")
        _ray_ws_bridge = RayWebSocketBridge.remote()
    return _ray_ws_bridge

