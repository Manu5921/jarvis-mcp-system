"""
Système de métriques pour Jarvis MCP
"""

import time
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict, deque

@dataclass
class MetricsCollector:
    """Collecteur de métriques pour l'orchestrateur"""
    
    def __init__(self):
        self.start_time = time.time()
        self.requests_count = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = deque(maxlen=1000)  # Garder les 1000 derniers
        self.agent_metrics = defaultdict(lambda: {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'total_time': 0.0
        })
        
    def record_request(self, agent_id: str, success: bool, response_time: float):
        """Enregistre une requête"""
        self.requests_count += 1
        self.response_times.append(response_time)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
        # Métriques par agent
        metrics = self.agent_metrics[agent_id]
        metrics['requests'] += 1
        metrics['total_time'] += response_time
        
        if success:
            metrics['successes'] += 1
        else:
            metrics['failures'] += 1
    
    async def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques"""
        
        uptime = time.time() - self.start_time
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0
        )
        
        success_rate = (
            self.successful_requests / self.requests_count
            if self.requests_count > 0 else 0
        )
        
        # Métriques par agent
        agent_stats = {}
        for agent_id, metrics in self.agent_metrics.items():
            agent_avg_time = (
                metrics['total_time'] / metrics['requests']
                if metrics['requests'] > 0 else 0
            )
            agent_success_rate = (
                metrics['successes'] / metrics['requests']
                if metrics['requests'] > 0 else 0
            )
            
            agent_stats[agent_id] = {
                'requests': metrics['requests'],
                'success_rate': agent_success_rate,
                'avg_response_time': agent_avg_time
            }
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.requests_count,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'agent_metrics': agent_stats,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }