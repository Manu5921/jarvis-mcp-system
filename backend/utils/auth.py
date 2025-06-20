"""
Authentication utilities for Jarvis MCP
JWT-based authentication and user management
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from db.database import get_database_session
from db.models import User

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 heures

# Utilitaires de hachage
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Modèles Pydantic
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_id: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe contre son hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hache un mot de passe"""
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Crée un token JWT"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

def verify_token(token: str) -> Optional[TokenData]:
    """Vérifie et décode un token JWT"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        
        if user_id is None:
            return None
            
        token_data = TokenData(user_id=user_id, username=username)
        return token_data
        
    except JWTError as e:
        logger.error(f"Erreur décodage JWT: {e}")
        return None

async def get_user_by_username(db, username: str) -> Optional[User]:
    """Récupère un utilisateur par nom d'utilisateur"""
    try:
        result = await db.execute(
            "SELECT * FROM users WHERE username = $1", username
        )
        user_data = result.fetchone()
        
        if user_data:
            return User(**dict(user_data))
        return None
        
    except Exception as e:
        logger.error(f"Erreur récupération utilisateur: {e}")
        return None

async def get_user_by_id(db, user_id: UUID) -> Optional[User]:
    """Récupère un utilisateur par ID"""
    try:
        result = await db.execute(
            "SELECT * FROM users WHERE id = $1", user_id
        )
        user_data = result.fetchone()
        
        if user_data:
            return User(**dict(user_data))
        return None
        
    except Exception as e:
        logger.error(f"Erreur récupération utilisateur par ID: {e}")
        return None

async def authenticate_user(db, username: str, password: str) -> Optional[User]:
    """Authentifie un utilisateur"""
    user = await get_user_by_username(db, username)
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    return user

async def create_user(db, user_data: UserCreate) -> User:
    """Crée un nouvel utilisateur"""
    try:
        # Vérifier si l'utilisateur existe déjà
        existing_user = await get_user_by_username(db, user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Hacher le mot de passe
        hashed_password = get_password_hash(user_data.password)
        
        # Créer l'utilisateur
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            is_active=True
        )
        
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        
        logger.info(f"Nouvel utilisateur créé: {user_data.username}")
        return new_user
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Erreur création utilisateur: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db = Depends(get_database_session)
) -> User:
    """Dependency pour obtenir l'utilisateur actuel depuis le token JWT"""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Extraire le token
        token = credentials.credentials
        
        # Vérifier le token
        token_data = verify_token(token)
        if token_data is None or token_data.user_id is None:
            raise credentials_exception
        
        # Récupérer l'utilisateur
        user = await get_user_by_id(db, UUID(token_data.user_id))
        if user is None:
            raise credentials_exception
        
        # Vérifier que l'utilisateur est actif
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        return user
        
    except ValueError:
        # Erreur de conversion UUID
        raise credentials_exception
    except Exception as e:
        logger.error(f"Erreur récupération utilisateur actuel: {e}")
        raise credentials_exception

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Dependency pour s'assurer que l'utilisateur est actif"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Inactive user"
        )
    return current_user

def create_user_token(user: User) -> Token:
    """Crée un token pour un utilisateur"""
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    token_data = {
        "sub": str(user.id),
        "username": user.username,
        "email": user.email
    }
    
    access_token = create_access_token(
        data=token_data, 
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # en secondes
        user_id=str(user.id)
    )

# Utilitaires pour les permissions
class PermissionChecker:
    """Vérificateur de permissions"""
    
    def __init__(self, required_permission: str):
        self.required_permission = required_permission
    
    def __call__(self, current_user: User = Depends(get_current_user)):
        # Pour l'instant, implémentation simple basée sur is_admin
        if self.required_permission == "admin" and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user

# Instances courantes
require_admin = PermissionChecker("admin")

# Middleware d'authentification pour WebSocket
async def authenticate_websocket_token(token: str, db) -> Optional[User]:
    """Authentifie un token pour WebSocket"""
    try:
        token_data = verify_token(token)
        if token_data and token_data.user_id:
            user = await get_user_by_id(db, UUID(token_data.user_id))
            if user and user.is_active:
                return user
        return None
    except Exception as e:
        logger.error(f"Erreur authentification WebSocket: {e}")
        return None

# Routes d'authentification (à intégrer dans main.py)
from fastapi import APIRouter

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

@auth_router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db = Depends(get_database_session)):
    """Inscription d'un nouvel utilisateur"""
    user = await create_user(db, user_data)
    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        created_at=user.created_at
    )

@auth_router.post("/login", response_model=Token)
async def login(user_data: UserLogin, db = Depends(get_database_session)):
    """Connexion utilisateur"""
    user = await authenticate_user(db, user_data.username, user_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Mettre à jour last_login
    user.last_login = datetime.now(timezone.utc)
    await db.commit()
    
    return create_user_token(user)

@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Informations de l'utilisateur actuel"""
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )

@auth_router.post("/refresh", response_model=Token)
async def refresh_token(current_user: User = Depends(get_current_user)):
    """Renouvellement du token"""
    return create_user_token(current_user)