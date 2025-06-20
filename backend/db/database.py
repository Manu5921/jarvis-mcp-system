"""
Database configuration and session management for Jarvis MCP
PostgreSQL with pgvector support and async operations
"""

import os
import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import text, event
from sqlalchemy.engine import Engine

from .models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gestionnaire de base de données pour Jarvis MCP"""
    
    def __init__(self, database_url: str, echo: bool = False):
        self.database_url = database_url
        self.echo = echo
        
        # Configuration du moteur async
        self.engine = create_async_engine(
            database_url,
            echo=echo,
            poolclass=NullPool,  # Pour éviter les problèmes avec pgvector
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "server_settings": {
                    "application_name": "jarvis_mcp",
                    "jit": "off",  # Optimisation pour pgvector
                }
            }
        )
        
        # Session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def init_database(self):
        """Initialise la base de données avec pgvector"""
        try:
            async with self.engine.begin() as conn:
                # Activer l'extension pgvector
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.info("Extension pgvector activée")
                
                # Créer toutes les tables
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Tables créées avec succès")
                
        except Exception as e:
            logger.error(f"Erreur initialisation base de données: {e}")
            raise
    
    async def drop_database(self):
        """Supprime toutes les tables (ATTENTION: perte de données)"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                logger.warning("Toutes les tables ont été supprimées")
        except Exception as e:
            logger.error(f"Erreur suppression tables: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Context manager pour sessions de base de données"""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """Ferme le moteur de base de données"""
        await self.engine.dispose()
        logger.info("Connexion base de données fermée")

# Instance globale du gestionnaire de base de données
database_manager: Optional[DatabaseManager] = None

def init_database_manager(database_url: str, echo: bool = False) -> DatabaseManager:
    """Initialise le gestionnaire de base de données global"""
    global database_manager
    database_manager = DatabaseManager(database_url, echo)
    return database_manager

async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection pour FastAPI"""
    if not database_manager:
        raise RuntimeError("Database manager not initialized")
    
    async with database_manager.get_session() as session:
        yield session

# Utilitaires pour les migrations et tests
async def create_test_database():
    """Crée une base de données de test en mémoire"""
    test_url = "postgresql+asyncpg://test:test@localhost:5432/jarvis_mcp_test"
    test_manager = DatabaseManager(test_url, echo=True)
    await test_manager.init_database()
    return test_manager

async def health_check(session: AsyncSession) -> dict:
    """Vérifie l'état de santé de la base de données"""
    try:
        # Test de connexion basique
        result = await session.execute(text("SELECT 1"))
        basic_ok = result.scalar() == 1
        
        # Test pgvector
        await session.execute(text("SELECT '[1,2,3]'::vector"))
        vector_ok = True
        
        # Statistiques de base
        stats_query = text("""
            SELECT 
                schemaname,
                tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes
            FROM pg_stat_user_tables 
            WHERE schemaname = 'public'
        """)
        stats_result = await session.execute(stats_query)
        table_stats = [dict(row._mapping) for row in stats_result]
        
        return {
            "status": "healthy" if (basic_ok and vector_ok) else "unhealthy",
            "connection": basic_ok,
            "pgvector": vector_ok,
            "table_statistics": table_stats,
            "database_url": database_manager.database_url if database_manager else None
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "connection": False,
            "pgvector": False
        }

# Event listeners pour optimisations PostgreSQL
@event.listens_for(Engine, "connect")
def set_postgresql_search_path(dbapi_connection, connection_record):
    """Configure le search path PostgreSQL"""
    with dbapi_connection.cursor() as cursor:
        # Optimisations pour pgvector
        cursor.execute("SET maintenance_work_mem = '512MB'")
        cursor.execute("SET max_parallel_maintenance_workers = 2")
        cursor.execute("SET effective_cache_size = '1GB'")

# Classes utilitaires pour les requêtes
class VectorSearchMixin:
    """Mixin pour recherches vectorielles optimisées"""
    
    @staticmethod
    async def similarity_search(
        session: AsyncSession,
        table_class,
        query_embedding: list,
        limit: int = 10,
        threshold: float = 0.8,
        filters: dict = None
    ):
        """Recherche par similarité vectorielle générique"""
        query = session.query(table_class)
        
        # Ajouter les filtres
        if filters:
            for key, value in filters.items():
                if hasattr(table_class, key):
                    query = query.filter(getattr(table_class, key) == value)
        
        # Recherche vectorielle
        query = query.filter(
            table_class.embedding.cosine_distance(query_embedding) < threshold
        ).order_by(
            table_class.embedding.cosine_distance(query_embedding)
        ).limit(limit)
        
        result = await session.execute(query)
        return result.scalars().all()

# Configuration des logs pour SQLAlchemy
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)