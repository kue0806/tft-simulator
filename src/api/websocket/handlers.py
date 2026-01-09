"""
WebSocket handlers for real-time communication.
"""

from typing import Dict, List
from fastapi import WebSocket


class WebSocketManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, game_id: str):
        """Connect a client to a game room."""
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
        self.active_connections[game_id].append(websocket)

    def disconnect(self, websocket: WebSocket, game_id: str):
        """Disconnect a client from a game room."""
        if game_id in self.active_connections:
            self.active_connections[game_id].remove(websocket)
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]

    async def broadcast(self, game_id: str, message: dict):
        """Broadcast message to all clients in a game room."""
        if game_id in self.active_connections:
            for connection in self.active_connections[game_id]:
                await connection.send_json(message)

    async def send_to_player(
        self, game_id: str, player_id: int, message: dict
    ):
        """Send message to specific player (placeholder)."""
        # In a full implementation, track player -> connection mapping
        await self.broadcast(game_id, message)


manager = WebSocketManager()
