"""OAuth token encryption at rest.

Encryption is enforced by API shape, not by convention: the Store exposes
only save_oauth_token/load_oauth_token, both of which require a TokenCipher.
There is no public path that persists a plaintext token.
"""

from __future__ import annotations

import json
from typing import Any

from cryptography.fernet import Fernet


class TokenCipher:
    def __init__(self, fernet_key: str) -> None:
        if not fernet_key:
            raise ValueError(
                "CALENDAI_FERNET_KEY is required to store OAuth tokens; generate one with "
                'python -c "from cryptography.fernet import Fernet; '
                'print(Fernet.generate_key().decode())"'
            )
        self._fernet = Fernet(fernet_key.encode())

    def encrypt(self, payload: dict[str, Any]) -> bytes:
        return self._fernet.encrypt(json.dumps(payload, sort_keys=True).encode())

    def decrypt(self, blob: bytes) -> dict[str, Any]:
        return json.loads(self._fernet.decrypt(blob))
