import os
import logging
from typing import Any, Dict, Optional, Tuple


from fastapi import FastAPI, Request, Response, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    LargeBinary,
    DateTime,
    func,
    select,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import boto3
from botocore.exceptions import ClientError

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://user:password@localhost:5432/securedb")
KMS_KEY_ID = os.getenv("KMS_KEY_ID", "alias/secure-data-key")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AUDIT_LOG_FILE = os.getenv("AUDIT_LOG_FILE", "audit.log")
API_TLS_ENABLED = True  # FastAPI should be run behind a TLS-enabled server (e.g., uvicorn with --ssl-keyfile/--ssl-certfile)

# Logging setup
logger = logging.getLogger("secure_data_layer")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(AUDIT_LOG_FILE)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# FastAPI app
app = FastAPI(
    title="Secure Data Encryption Layer",
    description="API for robust encryption of user data in transit and at rest.",
    version="1.0.0"
)

security = HTTPBearer()

# --- Key Management Integration (AWS KMS) ---

class KMSKeyManager:
    """
    Handles encryption key lifecycle using AWS KMS.
    """
    def __init__(self, key_id: str, region: str):
        self.kms = boto3.client("kms", region_name=region)
        self.key_id = key_id

    def generate_data_key(self) -> Tuple[bytes, bytes]:
        """
        Generates a new data key using KMS.
        Returns (plaintext_data_key, encrypted_data_key)
        """
        try:
            response = self.kms.generate_data_key(KeyId=self.key_id, KeySpec="AES_256")
            plaintext = response["Plaintext"]
            ciphertext = response["CiphertextBlob"]
            logger.info("Generated new data key via KMS (Ciphertext length: %d)", len(ciphertext))
            self._audit_log("generate_data_key", "success")
            return plaintext, ciphertext
        except ClientError as e:
            logger.error("KMS generate_data_key failed: %s", e)
            self._audit_log("generate_data_key", "failure")
            raise

    def decrypt_data_key(self, encrypted_data_key: bytes) -> bytes:
        """
        Decrypts an encrypted data key using KMS.
        """
        try:
            response = self.kms.decrypt(CiphertextBlob=encrypted_data_key)
            plaintext = response["Plaintext"]
            logger.info("Decrypted data key via KMS")
            self._audit_log("decrypt_data_key", "success")
            return plaintext
        except ClientError as e:
            logger.error("KMS decrypt_data_key failed: %s", e)
            self._audit_log("decrypt_data_key", "failure")
            raise

    def _audit_log(self, operation: str, status: str):
        logger.info(f"[KMS] {operation} - {status}")

# --- Audit Logging (No sensitive data exposed) ---

def audit_log(event: str, details: Optional[Dict[str, Any]] = None):
    """
    Logs encryption/decryption events without exposing sensitive data.
    """
    safe_details = {k: v for k, v in (details or {}).items() if k != "plaintext" and k != "ciphertext"}
    logger.info(f"[AUDIT] {event} | {safe_details}")

# --- Encryption/Decryption Utilities ---

class AESEncryptor:
    """
    AES-256-GCM encryption/decryption using data keys from KMS.
    """
    def __init__(self, key_manager: KMSKeyManager):
        self.key_manager = key_manager

    def encrypt(self, plaintext: bytes) -> Dict[str, bytes]:
        """
        Encrypts plaintext using a fresh data key and AES-GCM.
        Returns dict with ciphertext, nonce, tag, and encrypted_data_key.
        """
        data_key, encrypted_data_key = self.key_manager.generate_data_key()
        aesgcm = AESGCM(data_key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        audit_log("encrypt", {"length": len(plaintext)})
        return {
            "ciphertext": ciphertext,
            "nonce": nonce,
            "encrypted_data_key": encrypted_data_key
        }

    def decrypt(self, ciphertext: bytes, nonce: bytes, encrypted_data_key: bytes) -> bytes:
        """
        Decrypts ciphertext using the encrypted data key and AES-GCM.
        """
        data_key = self.key_manager.decrypt_data_key(encrypted_data_key)
        aesgcm = AESGCM(data_key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            audit_log("decrypt", {"length": len(ciphertext)})
            return plaintext
        except Exception as e:
            logger.error("Decryption failed: %s", e)
            audit_log("decrypt", {"status": "failure"})
            raise

# --- Database Models with Field-Level Encryption ---

class SensitiveData(Base):
    """
    Example model storing sensitive data encrypted at rest.
    """
    __tablename__ = "sensitive_data"

    id = Column(Integer, primary_key=True, index=True)
    # Encrypted fields
    encrypted_value = Column(LargeBinary, nullable=False)
    nonce = Column(LargeBinary, nullable=False)
    encrypted_data_key = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

# --- Dependency Injection ---

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_encryptor() -> AESEncryptor:
    key_manager = KMSKeyManager(KMS_KEY_ID, AWS_REGION)
    return AESEncryptor(key_manager)

# --- API Endpoints (All traffic must be over TLS) ---

@app.post("/store", status_code=201)
def store_sensitive_data(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
    encryptor: AESEncryptor = Depends(get_encryptor),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Stores sensitive data encrypted at rest.
    """
    try:
        value = payload.get("value")
        if not value or not isinstance(value, str):
            raise HTTPException(status_code=400, detail="Missing or invalid 'value' field.")
        # Encrypt the value
        encrypted = encryptor.encrypt(value.encode("utf-8"))
        record = SensitiveData(
            encrypted_value=encrypted["ciphertext"],
            nonce=encrypted["nonce"],
            encrypted_data_key=encrypted["encrypted_data_key"]
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        audit_log("store_sensitive_data", {"id": record.id})
        return {"id": record.id}
    except Exception as e:
        logger.error("Failed to store sensitive data: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/retrieve/{data_id}", status_code=200)
def retrieve_sensitive_data(
    data_id: int,
    db: Session = Depends(get_db),
    encryptor: AESEncryptor = Depends(get_encryptor),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Retrieves and decrypts sensitive data.
    """
    try:
        record = db.execute(
            select(SensitiveData).where(SensitiveData.id == data_id)
        ).scalar_one_or_none()
        if not record:
            raise HTTPException(status_code=404, detail="Data not found.")
        plaintext = encryptor.decrypt(
            ciphertext=record.encrypted_value,
            nonce=record.nonce,
            encrypted_data_key=record.encrypted_data_key
        )
        audit_log("retrieve_sensitive_data", {"id": data_id})
        return {"id": data_id, "value": plaintext.decode("utf-8")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve sensitive data: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")

# --- Middleware for Enforcing TLS (Optional, for defense-in-depth) ---

@app.middleware("http")
async def enforce_tls(request: Request, call_next):
    """
    Middleware to enforce TLS (defense-in-depth).
    """
    if API_TLS_ENABLED:
        if request.url.scheme != "https":
            logger.warning("Insecure request blocked (non-TLS).")
            return Response(
                content="TLS required.",
                status_code=status.HTTP_426_UPGRADE_REQUIRED
            )
    return await call_next(request)

# --- Database Initialization Utility ---

def init_db():
    """
    Initializes the database tables.
    """
    Base.metadata.create_all(bind=engine)

# --- Exported symbols ---

__all__ = [
    "app",
    "Base",
    "SensitiveData",
    "KMSKeyManager",
    "AESEncryptor",
    "audit_log",
    "init_db"
]