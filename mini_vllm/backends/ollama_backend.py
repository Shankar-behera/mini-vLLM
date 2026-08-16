"""
Thin async client for Ollama's /api/generate streaming endpoint.

This is deliberately minimal: send a prompt, get back an async stream of
ndjson chunks. It doesn't try to reproduce Ollama's own request queueing or
batching -- that's Ollama's job, running inside its own process. What we
control on our side is how many concurrent calls we allow out
(see real_engine.py), which is the honest boundary of what a black-box
completion API lets an external scheduler actually manage.

Note on the deprecated `context` field: older Ollama integrations drove
step-by-step generation by round-tripping a `context` array between calls.
That parameter is deprecated in current Ollama versions, so this client
doesn't use it -- it just consumes one streaming response per request from
start to finish.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Optional

import httpx


class OllamaConnectionError(RuntimeError):
    """Raised when the configured Ollama host can't be reached."""


class OllamaBackend:
    def __init__(self, host: str = "http://localhost:11434", timeout: float = 120.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    async def astream(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        client: Optional[httpx.AsyncClient] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Yields one dict per ndjson line from Ollama's streaming response.
        Each dict has at least a "response" (text piece, may be empty) and a
        "done" bool; the final chunk also carries "done_reason" and usage
        stats (eval_count, etc).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": 2048,
                "temperature": 0.7,
            },
        }

        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout)

        try:
            async with client.stream("POST", f"{self.host}/api/generate", json=payload) as resp:
                if resp.status_code == 404:
                    body = await resp.aread()
                    raise OllamaConnectionError(
                        f"model '{model}' not found on {self.host} "
                        f"(run `ollama pull {model}` first). Server said: {body.decode(errors='replace')[:200]}"
                    )
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise OllamaConnectionError(
                        f"Ollama returned HTTP {resp.status_code} from {self.host}: "
                        f"{body.decode(errors='replace')[:300]}"
                    )
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    yield json.loads(line)
        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"couldn't reach Ollama at {self.host} -- is `ollama serve` running?"
            ) from e
        finally:
            if owns_client:
                await client.aclose()
