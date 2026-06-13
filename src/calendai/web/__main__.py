"""Run the CalendAI web app:  python -m calendai.web

Provider/auth via environment:
  CALENDAI_PROVIDER=google (default) | fake
  CALENDAI_DEV_LOGIN=1     enable the dev login form (fake provider only)
  CALENDAI_HTTPS=1         set Secure on cookies (behind TLS)
"""

from __future__ import annotations

import os

import anthropic
import uvicorn

from calendai.core.config import get_settings
from calendai.web.app import create_app


def main() -> None:
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    app = create_app(
        settings=settings,
        agent_client=client,
        secure_cookies=os.environ.get("CALENDAI_HTTPS") == "1",
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
