"""
Jarvis MCP Database Models with pgvector support
PostgreSQL models for AI orchestration and memory management
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import uuid4, UUID

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, 
    JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class User(Base):
    """Utilisateur du système Jarvis MCP"""
    __tablename__ = "users"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relations
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"

class UserPreference(Base):
    """Préférences utilisateur pour l'orchestration IA"""
    __tablename__ = "user_preferences"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Préférences IA
    preferred_agent = Column(String(50), default="ollama", nullable=False)
    preferred_tone = Column(String(20), default="neutral", nullable=False)  # casual, formal, neutral
    preferred_language = Column(String(10), default="fr", nullable=False)
    auto_switch_enabled = Column(Boolean, default=True, nullable=False)
    parallel_processing = Column(Boolean, default=False, nullable=False)
    
    # Configuration personnalisée
    custom_rules = Column(JSONB, default=dict, nullable=False)
    agent_weights = Column(JSONB, default=dict, nullable=False)  # {"claude": 0.8, "ollama": 0.9}
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Relations
    user = relationship("User", back_populates="preferences")
    
    __table_args__ = (
        UniqueConstraint('user_id', name='uq_user_preferences'),
        CheckConstraint("preferred_tone IN ('casual', 'formal', 'neutral')", name='ck_tone'),
    )

class Session(Base):
    """Session de conversation avec l'orchestrateur MCP"""
    __tablename__ = "sessions"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Métadonnées session
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(String(20), default="active", nullable=False)  # active, paused, completed, error
    
    # Configuration session
    primary_agent = Column(String(50), nullable=True)
    agents_used = Column(JSONB, default=list, nullable=False)  # ["ollama", "claude"]
    session_config = Column(JSONB, default=dict, nullable=False)
    
    # Statistiques
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    avg_response_time = Column(Float, default=0.0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relations
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    
    # Index pour performance
    __table_args__ = (
        Index('ix_sessions_user_created', 'user_id', 'created_at'),
        Index('ix_sessions_status', 'status'),
    )

class Message(Base):
    """Messages dans une session avec embeddings vectoriels"""
    __tablename__ = "messages"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    
    # Contenu du message
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    agent_id = Column(String(50), nullable=True)  # ollama, claude, openai
    
    # Métadonnées
    message_type = Column(String(30), default="text", nullable=False)  # text, code, image, error
    category = Column(String(50), nullable=True)  # code, creative, analysis, general
    confidence_score = Column(Float, nullable=True)
    
    # Embeddings vectoriels pour similarité sémantique
    embedding = Column(Vector(384), nullable=True)  # sentence-transformers/all-MiniLM-L6-v2
    
    # Performance
    response_time = Column(Float, nullable=True)
    token_count = Column(Integer, default=0, nullable=False)
    
    # Métadonnées IA
    model_used = Column(String(100), nullable=True)
    temperature = Column(Float, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relations
    session = relationship("Session", back_populates="messages")
    feedback = relationship("Feedback", back_populates="message", cascade="all, delete-orphan")
    
    # Index pour recherche vectorielle et performance
    __table_args__ = (
        Index('ix_messages_session_created', 'session_id', 'created_at'),
        Index('ix_messages_role', 'role'),
        Index('ix_messages_agent', 'agent_id'),
        Index('ix_messages_category', 'category'),
        CheckConstraint("role IN ('user', 'assistant', 'system')", name='ck_message_role'),
    )

class Feedback(Base):
    """Feedback utilisateur pour amélioration continue"""
    __tablename__ = "feedback"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    message_id = Column(PG_UUID(as_uuid=True), ForeignKey("messages.id"), nullable=True)
    
    # Évaluation
    rating = Column(Integer, nullable=False)  # 1-5
    category = Column(String(30), nullable=False)  # accuracy, speed, relevance, helpfulness
    
    # Commentaires
    comment = Column(Text, nullable=True)
    suggestions = Column(Text, nullable=True)
    
    # Métadonnées
    agent_evaluated = Column(String(50), nullable=True)
    context_tags = Column(JSONB, default=list, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    # Relations
    user = relationship("User", back_populates="feedback")
    message = relationship("Message", back_populates="feedback")
    
    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name='ck_rating_range'),
        CheckConstraint("category IN ('accuracy', 'speed', 'relevance', 'helpfulness')", name='ck_feedback_category'),
        Index('ix_feedback_rating', 'rating'),
        Index('ix_feedback_agent', 'agent_evaluated'),
    )

class AgentMetrics(Base):
    """Métriques de performance des agents IA"""
    __tablename__ = "agent_metrics"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Identification agent
    agent_id = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=True)
    
    # Métriques de performance
    total_requests = Column(Integer, default=0, nullable=False)
    successful_requests = Column(Integer, default=0, nullable=False)
    failed_requests = Column(Integer, default=0, nullable=False)
    avg_response_time = Column(Float, default=0.0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    
    # Qualité
    avg_rating = Column(Float, nullable=True)
    feedback_count = Column(Integer, default=0, nullable=False)
    
    # Période de mesure
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_agent_metrics_period', 'agent_id', 'period_start', 'period_end'),
        UniqueConstraint('agent_id', 'period_start', 'period_end', name='uq_agent_period'),
    )

class MemoryVector(Base):
    """Mémoire vectorielle pour contexte sémantique"""
    __tablename__ = "memory_vectors"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Contenu
    content = Column(Text, nullable=False)
    content_type = Column(String(30), default="conversation", nullable=False)  # conversation, knowledge, preference
    
    # Embedding vectoriel
    embedding = Column(Vector(384), nullable=False)
    
    # Métadonnées
    importance_score = Column(Float, default=0.5, nullable=False)  # 0.0 - 1.0
    tags = Column(JSONB, default=list, nullable=False)
    source_session_id = Column(PG_UUID(as_uuid=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    last_accessed = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    access_count = Column(Integer, default=0, nullable=False)
    
    __table_args__ = (
        Index('ix_memory_user_importance', 'user_id', 'importance_score'),
        Index('ix_memory_type', 'content_type'),
        CheckConstraint("importance_score >= 0.0 AND importance_score <= 1.0", name='ck_importance_range'),
    )


# Fonctions utilitaires pour recherche vectorielle
def find_similar_messages(
    session: Session, 
    query_embedding: List[float], 
    limit: int = 5,
    threshold: float = 0.8
) -> List[Message]:
    """Trouve les messages similaires par embedding"""
    return session.query(Message).filter(
        Message.embedding.cosine_distance(query_embedding) < threshold
    ).order_by(
        Message.embedding.cosine_distance(query_embedding)
    ).limit(limit).all()

def find_relevant_memories(
    session: Session,
    user_id: UUID,
    query_embedding: List[float],
    limit: int = 10,
    threshold: float = 0.8
) -> List[MemoryVector]:
    """Trouve les souvenirs pertinents pour l'utilisateur"""
    return session.query(MemoryVector).filter(
        MemoryVector.user_id == user_id,
        MemoryVector.embedding.cosine_distance(query_embedding) < threshold
    ).order_by(
        MemoryVector.importance_score.desc(),
        MemoryVector.embedding.cosine_distance(query_embedding)
    ).limit(limit).all()