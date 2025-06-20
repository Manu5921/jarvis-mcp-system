"""
WebSocket Connection Manager for Jarvis MCP
Gestion des connexions WebSocket multi-clients
"""

import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class WebSocketConnection:
    """Représente une connexion WebSocket active"""
    
    def __init__(self, websocket: WebSocket, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.connected_at = datetime.now(timezone.utc)
        self.last_ping = datetime.now(timezone.utc)
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.channels: List[str] = []  # Canaux souscrits
        
    async def send_message(self, message: Dict[str, Any]):
        """Envoie un message via WebSocket"""
        try:
            await self.websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"Erreur envoi message WebSocket {self.client_id}: {e}")
            return False
    
    def update_ping(self):
        """Met à jour le timestamp du dernier ping"""
        self.last_ping = datetime.now(timezone.utc)
    
    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """Vérifie si la connexion est inactive"""
        elapsed = (datetime.now(timezone.utc) - self.last_ping).total_seconds()
        return elapsed > timeout_seconds

class WebSocketManager:
    """Gestionnaire des connexions WebSocket"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.channels: Dict[str, List[str]] = {}  # channel_name -> client_ids
        self.heartbeat_task: Optional[asyncio.Task] = None
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Ajoute une nouvelle connexion WebSocket"""
        await websocket.accept()
        
        connection = WebSocketConnection(websocket, client_id)
        self.connections[client_id] = connection
        
        logger.info(f"WebSocket connecté: {client_id}")
        
        # Envoyer message de confirmation
        await connection.send_message({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server_info": {
                "name": "Jarvis MCP",
                "version": "1.0.0"
            }
        })
    
    def disconnect(self, client_id: str):
        """Supprime une connexion WebSocket"""
        if client_id in self.connections:
            connection = self.connections[client_id]
            
            # Désabonner de tous les canaux
            for channel in connection.channels:
                self._unsubscribe_from_channel(client_id, channel)
            
            del self.connections[client_id]
            logger.info(f"WebSocket déconnecté: {client_id}")
    
    async def disconnect_all(self):
        """Déconnecte toutes les connexions WebSocket"""
        disconnect_tasks = []
        
        for client_id, connection in self.connections.items():
            try:
                await connection.websocket.close()
            except:
                pass  # Ignore les erreurs de fermeture
        
        self.connections.clear()
        self.channels.clear()
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        logger.info("Toutes les connexions WebSocket fermées")
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Envoie un message à un client spécifique"""
        if client_id in self.connections:
            return await self.connections[client_id].send_message(message)
        return False
    
    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any], exclude_client: Optional[str] = None):
        """Diffuse un message à tous les clients d'un canal"""
        if channel not in self.channels:
            return
        
        tasks = []
        for client_id in self.channels[channel]:
            if exclude_client and client_id == exclude_client:
                continue
            
            if client_id in self.connections:
                task = self.connections[client_id].send_message(message)
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failed_count = sum(1 for result in results if isinstance(result, Exception) or not result)
            
            if failed_count > 0:
                logger.warning(f"Échec envoi à {failed_count}/{len(tasks)} clients du canal {channel}")
    
    async def broadcast_to_all(self, message: Dict[str, Any], exclude_client: Optional[str] = None):
        """Diffuse un message à tous les clients connectés"""
        tasks = []
        for client_id, connection in self.connections.items():
            if exclude_client and client_id == exclude_client:
                continue
            
            task = connection.send_message(message)
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def subscribe_to_channel(self, client_id: str, channel: str):
        """Abonne un client à un canal"""
        if client_id not in self.connections:
            return False
        
        if channel not in self.channels:
            self.channels[channel] = []
        
        if client_id not in self.channels[channel]:
            self.channels[channel].append(client_id)
            self.connections[client_id].channels.append(channel)
            
            logger.info(f"Client {client_id} abonné au canal {channel}")
            return True
        
        return False
    
    def _unsubscribe_from_channel(self, client_id: str, channel: str):
        """Désabonne un client d'un canal"""
        if channel in self.channels and client_id in self.channels[channel]:
            self.channels[channel].remove(client_id)
            
            # Supprimer le canal s'il est vide
            if not self.channels[channel]:
                del self.channels[channel]
        
        if client_id in self.connections:
            connection = self.connections[client_id]
            if channel in connection.channels:
                connection.channels.remove(channel)
    
    def unsubscribe_from_channel(self, client_id: str, channel: str):
        """Désabonne un client d'un canal (public)"""
        self._unsubscribe_from_channel(client_id, channel)
        logger.info(f"Client {client_id} désabonné du canal {channel}")
    
    def get_connection_count(self) -> int:
        """Retourne le nombre de connexions actives"""
        return len(self.connections)
    
    def get_channel_count(self) -> int:
        """Retourne le nombre de canaux actifs"""
        return len(self.channels)
    
    def get_clients_in_channel(self, channel: str) -> List[str]:
        """Retourne la liste des clients dans un canal"""
        return self.channels.get(channel, []).copy()
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Retourne les informations d'un client"""
        if client_id not in self.connections:
            return None
        
        connection = self.connections[client_id]
        return {
            "client_id": client_id,
            "connected_at": connection.connected_at.isoformat(),
            "last_ping": connection.last_ping.isoformat(),
            "user_id": connection.user_id,
            "session_id": connection.session_id,
            "channels": connection.channels.copy(),
            "uptime_seconds": (datetime.now(timezone.utc) - connection.connected_at).total_seconds()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques globales"""
        active_connections = len(self.connections)
        total_channels = len(self.channels)
        
        # Calculer la distribution par canal
        channel_distribution = {
            channel: len(clients) 
            for channel, clients in self.channels.items()
        }
        
        # Connexions par âge
        now = datetime.now(timezone.utc)
        connection_ages = [
            (now - conn.connected_at).total_seconds()
            for conn in self.connections.values()
        ]
        
        avg_connection_age = sum(connection_ages) / len(connection_ages) if connection_ages else 0
        
        return {
            "active_connections": active_connections,
            "total_channels": total_channels,
            "channel_distribution": channel_distribution,
            "average_connection_age_seconds": avg_connection_age,
            "oldest_connection_seconds": max(connection_ages) if connection_ages else 0,
            "newest_connection_seconds": min(connection_ages) if connection_ages else 0
        }
    
    def cleanup_inactive_connections(self, timeout_seconds: int = 300):
        """Nettoie les connexions inactives"""
        stale_clients = []
        
        for client_id, connection in self.connections.items():
            if connection.is_stale(timeout_seconds):
                stale_clients.append(client_id)
        
        for client_id in stale_clients:
            logger.info(f"Nettoyage connexion inactive: {client_id}")
            self.disconnect(client_id)
        
        return len(stale_clients)
    
    def _start_heartbeat(self):
        """Démarre la tâche de heartbeat"""
        if self.heartbeat_task and not self.heartbeat_task.done():
            return
            
        async def heartbeat_loop():
            while True:
                try:
                    # Envoyer un ping à toutes les connexions
                    ping_message = {
                        "type": "ping",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Collecter les tâches de ping
                    ping_tasks = []
                    for connection in self.connections.values():
                        task = connection.send_message(ping_message)
                        ping_tasks.append(task)
                    
                    if ping_tasks:
                        results = await asyncio.gather(*ping_tasks, return_exceptions=True)
                        
                        # Identifier les connexions qui ont échoué
                        failed_clients = []
                        for i, (client_id, result) in enumerate(zip(self.connections.keys(), results)):
                            if isinstance(result, Exception) or not result:
                                failed_clients.append(client_id)
                        
                        # Nettoyer les connexions échouées
                        for client_id in failed_clients:
                            self.disconnect(client_id)
                    
                    # Attendre 30 secondes avant le prochain ping
                    await asyncio.sleep(30)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Erreur heartbeat WebSocket: {e}")
                    await asyncio.sleep(5)
        
        try:
            self.heartbeat_task = asyncio.create_task(heartbeat_loop())
        except RuntimeError:
            # No event loop running, will start during application startup
            pass
    
    async def handle_client_message(self, client_id: str, message: Dict[str, Any]):
        """Traite les messages spéciaux des clients"""
        message_type = message.get("type")
        
        if message_type == "pong":
            # Répondre au pong en mettant à jour le timestamp
            if client_id in self.connections:
                self.connections[client_id].update_ping()
        
        elif message_type == "subscribe":
            # S'abonner à un canal
            channel = message.get("channel")
            if channel:
                success = self.subscribe_to_channel(client_id, channel)
                await self.send_to_client(client_id, {
                    "type": "subscription_result",
                    "channel": channel,
                    "success": success
                })
        
        elif message_type == "unsubscribe":
            # Se désabonner d'un canal
            channel = message.get("channel")
            if channel:
                self.unsubscribe_from_channel(client_id, channel)
                await self.send_to_client(client_id, {
                    "type": "unsubscription_result",
                    "channel": channel,
                    "success": True
                })
        
        elif message_type == "authenticate":
            # Authentifier le client
            user_id = message.get("user_id")
            session_id = message.get("session_id")
            
            if client_id in self.connections:
                connection = self.connections[client_id]
                connection.user_id = user_id
                connection.session_id = session_id
                
                await self.send_to_client(client_id, {
                    "type": "authentication_result",
                    "success": True,
                    "user_id": user_id
                })