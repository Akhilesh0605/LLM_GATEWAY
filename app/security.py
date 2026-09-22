import hashlib
import secrets
from cryptography.fernet import Fernet
from app.config import get_settings

# Initialize Fernet suite with master encryption key
settings = get_settings()
fernet = Fernet(settings.ENCRYPTION_KEY.encode())


def encrypt_api_key(raw_key: str) -> str:
    """Encrypts user's third-party API key for DB storage."""
    return fernet.encrypt(raw_key.encode()).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypts stored API key back to plaintext for request execution."""
    return fernet.decrypt(encrypted_key.encode()).decode()


def generate_gateway_key() -> tuple[str, str]:
    """
    Generates a secure gateway key.
    Returns: (raw_key, hashed_key)
    Raw key is shown ONCE to the user. Hashed key is stored in DB.
    """
    raw_key = f"gw_live_{secrets.token_hex(16)}"
    hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()
    return raw_key, hashed_key


def hash_gateway_key(raw_key: str) -> str:
    """Hashes an incoming X-Gateway-Key header for DB lookup."""
    return hashlib.sha256(raw_key.encode()).hexdigest()