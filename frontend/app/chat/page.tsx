'use client'

import { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import Link from 'next/link'

interface Message {
  id: string
  content: string
  sender: 'user' | 'assistant'
  timestamp: Date
  agent?: string
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedAgent, setSelectedAgent] = useState<string>('auto')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Message de bienvenue
    const welcomeMessage: Message = {
      id: '0',
      content: 'Bonjour ! Je suis Jarvis MCP, votre orchestrateur IA intelligent. Je peux coordonner plusieurs agents IA pour vous donner les meilleures réponses. Comment puis-je vous aider aujourd\'hui ?',
      sender: 'assistant',
      timestamp: new Date(),
      agent: 'Jarvis MCP'
    }
    setMessages([welcomeMessage])
  }, [])

  const sendMessage = async () => {
    if (!input.trim() || loading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      sender: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      // Simulation d'appel API
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `Je vais traiter votre demande "${input}" en utilisant le meilleur agent disponible. ${
          selectedAgent === 'claude' ? 'Utilisation de Claude pour cette réponse spécialisée.' :
          selectedAgent === 'ollama' ? 'Utilisation d\'Ollama pour une réponse locale rapide.' :
          selectedAgent === 'openai' ? 'Utilisation d\'OpenAI pour une réponse créative.' :
          'Sélection automatique de l\'agent le plus adapté.'
        }`,
        sender: 'assistant',
        timestamp: new Date(),
        agent: selectedAgent === 'auto' ? 'Claude' : selectedAgent
      }

      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      console.error('Erreur lors de l\'envoi du message:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Désolé, une erreur s\'est produite. Veuillez réessayer.',
        sender: 'assistant',
        timestamp: new Date(),
        agent: 'Système'
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-jarvis-50 to-jarvis-100 dark:from-gray-900 dark:to-gray-800 flex flex-col">
      {/* Header */}
      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700 p-4">
        <div className="container mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-bold jarvis-gradient-text">Chat Jarvis MCP</h1>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600 dark:text-gray-300">En ligne</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Sélecteur d'agent */}
            <select 
              value={selectedAgent}
              onChange={(e) => setSelectedAgent(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md bg-white dark:bg-gray-700 dark:border-gray-600"
            >
              <option value="auto">🤖 Auto (Intelligent)</option>
              <option value="claude">🧠 Claude (Code & Analyse)</option>
              <option value="ollama">⚡ Ollama (Local & Rapide)</option>
              <option value="openai">✨ OpenAI (Créatif)</option>
            </select>
            
            <Link href="/dashboard">
              <Button variant="outline">Dashboard</Button>
            </Link>
            <Link href="/">
              <Button variant="outline">Accueil</Button>
            </Link>
          </div>
        </div>
      </div>

      {/* Zone de messages */}
      <div className="flex-1 container mx-auto px-4 py-6 overflow-hidden flex flex-col">
        <div className="flex-1 overflow-y-auto space-y-4 custom-scrollbar">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`message-bubble ${message.sender}`}>
                {message.sender === 'assistant' && message.agent && (
                  <div className="text-xs font-medium text-jarvis-600 dark:text-jarvis-400 mb-1">
                    {message.agent}
                  </div>
                )}
                <div className="whitespace-pre-wrap">{message.content}</div>
                <div className="text-xs opacity-70 mt-2">
                  {message.timestamp.toLocaleTimeString('fr-FR', { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                  })}
                </div>
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="flex justify-start">
              <div className="message-bubble assistant">
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Zone de saisie */}
        <div className="mt-6 p-4 bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="flex space-x-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Tapez votre message... (Entrée pour envoyer, Maj+Entrée pour nouvelle ligne)"
              className="flex-1 resize-none rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 px-3 py-2 text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-jarvis-500 focus:border-transparent"
              rows={3}
              disabled={loading}
            />
            <Button 
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="jarvis-gradient text-white"
            >
              {loading ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              ) : (
                'Envoyer'
              )}
            </Button>
          </div>
          
          <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
            <span>
              Agent sélectionné: <strong>{
                selectedAgent === 'auto' ? 'Sélection automatique' :
                selectedAgent === 'claude' ? 'Claude (Analyse)' :
                selectedAgent === 'ollama' ? 'Ollama (Local)' :
                'OpenAI (Créatif)'
              }</strong>
            </span>
            <span>{input.length}/2000 caractères</span>
          </div>
        </div>
      </div>
    </div>
  )
}