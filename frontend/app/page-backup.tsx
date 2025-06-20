'use client'

import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { useEffect, useState } from 'react'

function AnimatedBackground() {
  const [particles, setParticles] = useState<Array<{left: number, top: number, delay: number, duration: number}>>([])

  useEffect(() => {
    // Generate particles only on client side to avoid hydration mismatch
    const newParticles = [...Array(50)].map(() => ({
      left: Math.random() * 100,
      top: Math.random() * 100,
      delay: Math.random() * 3,
      duration: 2 + Math.random() * 3,
    }))
    setParticles(newParticles)
  }, [])

  return (
    <div className="absolute inset-0 -z-10 overflow-hidden">
      {/* Gradient mesh background */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_800px_600px_at_50%_0%,rgba(120,119,198,0.3),transparent)]" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_400px_300px_at_80%_70%,rgba(147,51,234,0.15),transparent)]" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_600px_400px_at_20%_80%,rgba(59,130,246,0.15),transparent)]" />
      
      {/* Floating particles */}
      <div className="absolute inset-0">
        {particles.map((particle, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-blue-400/20 rounded-full animate-pulse"
            style={{
              left: `${particle.left}%`,
              top: `${particle.top}%`,
              animationDelay: `${particle.delay}s`,
              animationDuration: `${particle.duration}s`,
            }}
          />
        ))}
      </div>
    </div>
  )
}

function FeatureCard({ 
  icon, 
  title, 
  description, 
  delay = 0 
}: { 
  icon: string; 
  title: string; 
  description: string; 
  delay?: number;
}) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), delay * 200)
    return () => clearTimeout(timer)
  }, [delay])

  return (
    <div className={`
      group relative overflow-hidden rounded-2xl bg-white/10 backdrop-blur-sm border border-white/20 p-8
      transition-all duration-700 hover:scale-105 hover:bg-white/15 hover:border-white/30
      ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}
    `}>
      {/* Gradient border effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 via-purple-500/20 to-teal-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      
      {/* Content */}
      <div className="relative z-10">
        <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-300">
          {icon}
        </div>
        <h3 className="text-xl font-bold text-white mb-3 group-hover:text-blue-200 transition-colors">
          {title}
        </h3>
        <p className="text-gray-300 group-hover:text-gray-200 transition-colors">
          {description}
        </p>
      </div>
      
      {/* Hover glow effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/0 via-purple-500/0 to-teal-500/0 group-hover:from-blue-500/5 group-hover:via-purple-500/5 group-hover:to-teal-500/5 transition-all duration-500" />
    </div>
  )
}

function StatusIndicator() {
  const [status, setStatus] = useState<'connecting' | 'online' | 'offline'>('connecting')

  useEffect(() => {
    // Simulate connection check
    const timer = setTimeout(() => setStatus('online'), 1000)
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/20">
      <div className={`w-2 h-2 rounded-full transition-all duration-500 ${
        status === 'connecting' ? 'bg-yellow-400 animate-pulse' :
        status === 'online' ? 'bg-green-400 animate-pulse' :
        'bg-red-400'
      }`} />
      <span className="text-sm text-white/90 font-medium">
        {status === 'connecting' ? 'Connexion...' :
         status === 'online' ? 'Système opérationnel' :
         'Hors ligne'}
      </span>
    </div>
  )
}

export default function Home() {
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    setIsLoaded(true)
  }, [])

  return (
    <main className="relative min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 overflow-hidden">
      <AnimatedBackground />
      
      <div className="relative z-10 container mx-auto px-4 py-16">
        {/* Header with status */}
        <div className="flex justify-between items-center mb-16">
          <div className={`transition-all duration-1000 ${isLoaded ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-8'}`}>
            <StatusIndicator />
          </div>
          <div className={`transition-all duration-1000 delay-200 ${isLoaded ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-8'}`}>
            <div className="text-right text-sm text-white/60">
              Version 1.0.0-alpha
            </div>
          </div>
        </div>

        {/* Hero Section */}
        <div className="text-center mb-20">
          <div className={`transition-all duration-1000 delay-300 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
            <h1 className="text-7xl lg:text-8xl font-bold mb-6 relative">
              <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-teal-400 bg-clip-text text-transparent animate-gradient-x">
                Jarvis MCP
              </span>
              {/* Animated underline */}
              <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-0 h-1 bg-gradient-to-r from-blue-400 to-purple-400 animate-expand-width" />
            </h1>
          </div>
          
          <div className={`transition-all duration-1000 delay-500 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
            <p className="text-2xl lg:text-3xl text-white/90 mb-6 font-light">
              Multi-Channel Processor pour
            </p>
            <p className="text-2xl lg:text-3xl bg-gradient-to-r from-blue-300 to-purple-300 bg-clip-text text-transparent font-semibold mb-8">
              orchestration IA intelligente
            </p>
          </div>
          
          <div className={`transition-all duration-1000 delay-700 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
            <p className="text-lg text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
              Coordonnez plusieurs agents IA (Ollama, Claude, OpenAI) pour des réponses 
              optimisées selon le contexte et vos préférences. Une interface moderne 
              pour l'avenir de l'intelligence artificielle.
            </p>
          </div>
          
          {/* Action Buttons */}
          <div className={`transition-all duration-1000 delay-900 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
              <Link href="/dashboard">
                <Button 
                  size="lg" 
                  className="group relative overflow-hidden bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white border-0 px-8 py-4 text-lg font-semibold transition-all duration-300 hover:scale-105 hover:shadow-2xl hover:shadow-blue-500/25"
                >
                  <span className="relative z-10 flex items-center gap-2">
                    🚀 Accéder au Dashboard
                  </span>
                  <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/10 to-white/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
                </Button>
              </Link>
              <Link href="/chat">
                <Button 
                  size="lg" 
                  variant="outline"
                  className="group relative overflow-hidden border-2 border-white/30 text-white hover:bg-white/10 hover:border-white/50 px-8 py-4 text-lg font-semibold transition-all duration-300 hover:scale-105 backdrop-blur-sm"
                >
                  <span className="relative z-10 flex items-center gap-2">
                    💬 Commencer à chatter
                  </span>
                </Button>
              </Link>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mt-20">
          <FeatureCard
            icon="🧠"
            title="Orchestration Intelligente"
            description="Routage automatique vers l'agent IA le plus adapté selon le contexte et la complexité de votre demande"
            delay={0}
          />
          
          <FeatureCard
            icon="🔌"
            title="Multi-Agents"
            description="Ollama local pour la rapidité, Claude pour le code et l'analyse, OpenAI pour la créativité"
            delay={1}
          />
          
          <FeatureCard
            icon="💾"
            title="Mémoire Vectorielle"
            description="PostgreSQL + pgvector pour un contexte sémantique intelligent et des conversations persistantes"
            delay={2}
          />
        </div>

        {/* Performance Metrics */}
        <div className={`transition-all duration-1000 delay-1100 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
          <div className="mt-20 text-center">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              {[
                { label: 'Agents Actifs', value: '3', suffix: '' },
                { label: 'Temps de Réponse', value: '<100', suffix: 'ms' },
                { label: 'Taux de Succès', value: '99.9', suffix: '%' },
                { label: 'Conversations', value: '∞', suffix: '' },
              ].map((metric, index) => (
                <div key={index} className="text-center">
                  <div className="text-3xl lg:text-4xl font-bold text-white mb-2">
                    {metric.value}
                    <span className="text-blue-400">{metric.suffix}</span>
                  </div>
                  <div className="text-sm text-gray-400 uppercase tracking-wider">
                    {metric.label}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-slate-900 to-transparent" />
    </main>
  )
}