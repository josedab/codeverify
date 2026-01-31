"""Encryption utilities for sensitive data."""

import base64
import os
import secrets
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from codeverify_api.config import settings


def _get_encryption_key() -> bytes:
    """Derive encryption key from JWT secret.
    
    Uses PBKDF2 to derive a Fernet-compatible key from the JWT secret.
    """
    # Use JWT secret as base for encryption key
    password = settings.JWT_SECRET.encode()
    
    # Use a fixed salt derived from the secret itself for consistency
    # In production, you might want to store this separately
    salt = base64.urlsafe_b64decode(
        base64.urlsafe_b64encode(password[:16].ljust(16, b'\x00'))
    )
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return key


def _get_fernet() -> Fernet:
    """Get Fernet instance for encryption/decryption."""
    return Fernet(_get_encryption_key())


def encrypt_token(token: str) -> str:
    """Encrypt a sensitive token for storage.
    
    Args:
        token: The plaintext token to encrypt
        
    Returns:
        Base64-encoded encrypted token
    """
    if not token:
        return ""
    
    fernet = _get_fernet()
    encrypted = fernet.encrypt(token.encode())
    return base64.urlsafe_b64encode(encrypted).decode()


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt a stored token.
    
    Args:
        encrypted_token: Base64-encoded encrypted token
        
    Returns:
        Decrypted plaintext token
        
    Raises:
        ValueError: If decryption fails
    """
    if not encrypted_token:
        return ""
    
    try:
        fernet = _get_fernet()
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_token.encode())
        decrypted = fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    except Exception as e:
        raise ValueError(f"Failed to decrypt token: {e}")


def is_encrypted(value: str) -> bool:
    """Check if a value appears to be encrypted.
    
    This is a heuristic check - encrypted tokens are longer
    and have a specific format.
    """
    if not value:
        return False
    
    # Encrypted tokens are significantly longer than GitHub tokens
    # and are base64 encoded
    if len(value) < 100:
        return False
    
    try:
        base64.urlsafe_b64decode(value.encode())
        return True
    except Exception:
        return False
