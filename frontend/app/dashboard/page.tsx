'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import Link from 'next/link'

interface AgentStatus {
  name: string
  status: 'online' | 'offline' | 'busy'
  requests: number
  response_time: number
}

interface SystemMetrics {
  uptime: number
  total_requests: number
  success_rate: number
  agents: Record<string, AgentStatus>
}

export default function Dashboard() {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch('/api/metrics')
        if (response.ok) {
          const data = await response.json()
          setMetrics(data)
        }
      } catch (error) {
        console.error('Erreur lors du chargement des métriques:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchMetrics()
    const interval = setInterval(fetchMetrics, 10000) // Actualiser toutes les 10s

    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-jarvis-50 to-jarvis-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-jarvis-500 mx-auto"></div>
          <p className="mt-4 text-lg text-gray-600 dark:text-gray-300">Chargement du dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-jarvis-50 to-jarvis-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold jarvis-gradient-text mb-2">Dashboard Jarvis MCP</h1>
            <p className="text-gray-600 dark:text-gray-300">Surveillance et contrôle de vos agents IA</p>
          </div>
          <div className="flex gap-4">
            <Link href="/chat">
              <Button size="lg" className="jarvis-gradient text-white">
                Nouveau Chat
              </Button>
            </Link>
            <Link href="/">
              <Button size="lg" variant="outline">
                Accueil
              </Button>
            </Link>
          </div>
        </div>

        {/* Métriques système */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="metric-card">
            <div className="text-center">
              <div className="metric-value">{metrics?.total_requests || 0}</div>
              <div className="metric-label">Requêtes totales</div>
            </div>
          </div>
          
          <div className="metric-card">
            <div className="text-center">
              <div className="metric-value">{Math.round((metrics?.success_rate || 0) * 100)}%</div>
              <div className="metric-label">Taux de succès</div>
            </div>
          </div>
          
          <div className="metric-card">
            <div className="text-center">
              <div className="metric-value">{Math.floor((metrics?.uptime || 0) / 60)}m</div>
              <div className="metric-label">Temps de fonctionnement</div>
            </div>
          </div>
          
          <div className="metric-card">
            <div className="text-center">
              <div className="metric-value">{Object.keys(metrics?.agents || {}).length}</div>
              <div className="metric-label">Agents actifs</div>
            </div>
          </div>
        </div>

        {/* État des agents */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="metric-card">
            <h2 className="text-2xl font-semibold mb-6">État des Agents IA</h2>
            <div className="space-y-4">
              {[
                { name: 'Ollama Local', status: 'online', description: 'Modèle local rapide' },
                { name: 'Claude', status: 'online', description: 'Expert en code et analyse' },
                { name: 'OpenAI', status: 'online', description: 'Créativité et général' }
              ].map((agent, index) => (
                <div key={index} className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${
                      agent.status === 'online' ? 'bg-green-500' : 
                      agent.status === 'busy' ? 'bg-yellow-500' : 'bg-red-500'
                    }`}></div>
                    <div>
                      <div className="font-medium">{agent.name}</div>
                      <div className="text-sm text-gray-500">{agent.description}</div>
                    </div>
                  </div>
                  <span className={`agent-status ${agent.status}`}>
                    {agent.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="metric-card">
            <h2 className="text-2xl font-semibold mb-6">Activité Récente</h2>
            <div className="space-y-3">
              {[
                { time: '14:32', user: 'Utilisateur', action: 'Nouvelle conversation démarrée', agent: 'Claude' },
                { time: '14:28', user: 'Système', action: 'Agent Ollama connecté', agent: 'Ollama' },
                { time: '14:25', user: 'Utilisateur', action: 'Question sur React posée', agent: 'Claude' },
                { time: '14:20', user: 'Système', action: 'Synchronisation des métriques', agent: 'Système' }
              ].map((activity, index) => (
                <div key={index} className="flex items-center space-x-3 text-sm">
                  <span className="text-gray-500 w-12">{activity.time}</span>
                  <span className="font-medium">{activity.user}</span>
                  <span className="text-gray-600 dark:text-gray-300 flex-1">{activity.action}</span>
                  <span className="text-jarvis-600 font-medium">{activity.agent}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Actions rapides */}
        <div className="mt-8">
          <div className="metric-card">
            <h2 className="text-2xl font-semibold mb-6">Actions Rapides</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Button variant="outline" className="h-24 flex-col">
                <span className="text-2xl mb-2">🔄</span>
                Redémarrer Agents
              </Button>
              <Button variant="outline" className="h-24 flex-col">
                <span className="text-2xl mb-2">📊</span>
                Exporter Métriques
              </Button>
              <Button variant="outline" className="h-24 flex-col">
                <span className="text-2xl mb-2">⚙️</span>
                Configuration
              </Button>
              <Button variant="outline" className="h-24 flex-col">
                <span className="text-2xl mb-2">📝</span>
                Logs Système
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}