"""
Embedding Management for Jarvis MCP
Gestion des embeddings vectoriels pour la mémoire sémantique
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Gestionnaire d'embeddings pour la mémoire vectorielle"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.vector_dimensions = config.get("vector_dimensions", 384)
        self.model: Optional[SentenceTransformer] = None
        
        # Cache des embeddings récents
        self.embedding_cache: Dict[str, List[float]] = {}
        self.cache_max_size = 1000
        
    async def initialize(self):
        """Initialise le modèle d'embeddings"""
        try:
            # Charger le modèle en arrière-plan pour éviter de bloquer
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.model_name)
            )
            
            logger.info(f"Modèle d'embeddings initialisé: {self.model_name}")
            logger.info(f"Dimensions vectorielles: {self.vector_dimensions}")
            
            # Test du modèle
            test_embedding = await self.create_embedding("Test")
            if len(test_embedding) != self.vector_dimensions:
                logger.warning(
                    f"Dimensions attendues: {self.vector_dimensions}, "
                    f"reçues: {len(test_embedding)}"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur initialisation embeddings: {e}")
            return False
    
    async def create_embedding(self, text: str) -> List[float]:
        """Crée un embedding pour un texte donné"""
        if not self.model:
            raise RuntimeError("Modèle d'embeddings non initialisé")
        
        # Vérifier le cache
        cache_key = self._get_cache_key(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Nettoyer et préparer le texte
            cleaned_text = self._preprocess_text(text)
            
            # Générer l'embedding
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode(cleaned_text, convert_to_numpy=True)
            )
            
            # Convertir en liste et normaliser
            embedding_list = embedding.tolist()
            normalized_embedding = self._normalize_embedding(embedding_list)
            
            # Ajouter au cache
            self._add_to_cache(cache_key, normalized_embedding)
            
            return normalized_embedding
            
        except Exception as e:
            logger.error(f"Erreur création embedding: {e}")
            # Retourner un embedding zéro en cas d'erreur
            return [0.0] * self.vector_dimensions
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Crée des embeddings pour une liste de textes (optimisé)"""
        if not self.model:
            raise RuntimeError("Modèle d'embeddings non initialisé")
        
        # Séparer les textes cachés des nouveaux
        cached_embeddings = {}
        new_texts = []
        new_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                cached_embeddings[i] = self.embedding_cache[cache_key]
            else:
                new_texts.append(self._preprocess_text(text))
                new_indices.append(i)
        
        # Générer les nouveaux embeddings
        new_embeddings = []
        if new_texts:
            try:
                loop = asyncio.get_event_loop()
                embeddings_array = await loop.run_in_executor(
                    None,
                    lambda: self.model.encode(new_texts, convert_to_numpy=True)
                )
                
                new_embeddings = [
                    self._normalize_embedding(emb.tolist()) 
                    for emb in embeddings_array
                ]
                
                # Ajouter au cache
                for i, (text, embedding) in enumerate(zip(new_texts, new_embeddings)):
                    cache_key = self._get_cache_key(text)
                    self._add_to_cache(cache_key, embedding)
                
            except Exception as e:
                logger.error(f"Erreur création embeddings batch: {e}")
                new_embeddings = [[0.0] * self.vector_dimensions] * len(new_texts)
        
        # Recombiner les résultats
        result = [None] * len(texts)
        
        # Ajouter les embeddings cachés
        for i, embedding in cached_embeddings.items():
            result[i] = embedding
        
        # Ajouter les nouveaux embeddings
        for i, embedding in zip(new_indices, new_embeddings):
            result[i] = embedding
        
        return result
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calcule la similarité cosinus entre deux embeddings"""
        try:
            # Convertir en numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calcul similarité cosinus
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # S'assurer que le résultat est dans [-1, 1]
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Erreur calcul similarité: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]],
        threshold: float = 0.8
    ) -> List[Tuple[int, float]]:
        """Trouve les embeddings les plus similaires"""
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            if similarity >= threshold:
                similarities.append((i, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def _preprocess_text(self, text: str) -> str:
        """Prétraite le texte avant embedding"""
        # Nettoyer le texte
        cleaned = text.strip()
        
        # Limiter la longueur (les modèles ont des limites)
        max_length = 512  # Limite typique pour les modèles sentence-transformers
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
        
        return cleaned
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalise un embedding (norme L2)"""
        try:
            vec = np.array(embedding)
            norm = np.linalg.norm(vec)
            
            if norm == 0:
                return embedding
            
            normalized = vec / norm
            return normalized.tolist()
            
        except Exception as e:
            logger.error(f"Erreur normalisation embedding: {e}")
            return embedding
    
    def _get_cache_key(self, text: str) -> str:
        """Génère une clé de cache pour un texte"""
        # Utiliser un hash pour économiser la mémoire
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _add_to_cache(self, key: str, embedding: List[float]):
        """Ajoute un embedding au cache avec gestion de la taille"""
        # Nettoyer le cache s'il est trop grand
        if len(self.embedding_cache) >= self.cache_max_size:
            # Supprimer les plus anciens (FIFO simple)
            oldest_keys = list(self.embedding_cache.keys())[:100]
            for old_key in oldest_keys:
                del self.embedding_cache[old_key]
        
        self.embedding_cache[key] = embedding
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache"""
        return {
            "cache_size": len(self.embedding_cache),
            "cache_max_size": self.cache_max_size,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "model_name": self.model_name,
            "vector_dimensions": self.vector_dimensions
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calcule le taux de succès du cache (approximatif)"""
        # Pour une implémentation complète, il faudrait tracker les hits/misses
        # Ici on retourne juste un ratio basé sur la taille du cache
        return min(len(self.embedding_cache) / self.cache_max_size, 1.0)
    
    def clear_cache(self):
        """Vide le cache des embeddings"""
        self.embedding_cache.clear()
        logger.info("Cache d'embeddings vidé")
    
    async def cleanup(self):
        """Nettoie les ressources"""
        self.clear_cache()
        self.model = None
        logger.info("Gestionnaire d'embeddings nettoyé")

# Fonctions utilitaires pour l'intégration avec la base de données
async def create_memory_embedding(
    embedding_manager: EmbeddingManager, 
    content: str, 
    content_type: str = "conversation"
) -> Tuple[List[float], Dict[str, Any]]:
    """Crée un embedding pour la mémoire avec métadonnées"""
    
    # Créer l'embedding
    embedding = await embedding_manager.create_embedding(content)
    
    # Générer des métadonnées
    metadata = {
        "content_length": len(content),
        "content_type": content_type,
        "created_at": asyncio.get_event_loop().time(),
        "embedding_model": embedding_manager.model_name
    }
    
    return embedding, metadata

async def search_similar_memories(
    embedding_manager: EmbeddingManager,
    query: str,
    memory_embeddings: List[Tuple[str, List[float]]],  # (content, embedding)
    limit: int = 5,
    threshold: float = 0.8
) -> List[Tuple[str, float]]:
    """Recherche les souvenirs similaires à une requête"""
    
    # Créer l'embedding de la requête
    query_embedding = await embedding_manager.create_embedding(query)
    
    # Extraire seulement les embeddings
    embeddings = [emb for _, emb in memory_embeddings]
    
    # Trouver les plus similaires
    similarities = embedding_manager.find_most_similar(
        query_embedding, embeddings, threshold
    )
    
    # Retourner les contenus avec leurs scores
    results = []
    for idx, score in similarities[:limit]:
        content = memory_embeddings[idx][0]
        results.append((content, score))
    
    return results