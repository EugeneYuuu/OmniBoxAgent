"""AES-256-GCM decryption utility (compatible with Java AesEncryptionUtil).

Format: Base64( IV[12B] + CipherText + GCMTag[16B] )
"""

from __future__ import annotations

import base64
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_IV_LENGTH = 12
_GCM_TAG_LENGTH = 16  # 128 bits = 16 bytes


def decrypt_api_key(encrypted: str, key_base64: str) -> str:
    """Decrypt an AES-256-GCM encrypted API key.

    Args:
        encrypted: Base64-encoded ciphertext (IV + ciphertext + tag).
        key_base64: Base64-encoded 32-byte AES key.

    Returns:
        Decrypted plaintext string.
    """
    if not encrypted or not key_base64:
        return ""

    key = base64.b64decode(key_base64)
    if len(key) != 32:
        raise ValueError(f"AES key must be 32 bytes, got {len(key)}")

    combined = base64.b64decode(encrypted)
    iv = combined[:_IV_LENGTH]
    ciphertext_with_tag = combined[_IV_LENGTH:]

    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(iv, ciphertext_with_tag, None)
    return plaintext.decode("utf-8")
