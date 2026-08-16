import json

import httpx
import pytest

from mini_vllm.backends.ollama_backend import OllamaBackend, OllamaConnectionError


def _ndjson_response(chunks):
    body = "\n".join(json.dumps(c) for c in chunks) + "\n"
    return httpx.Response(200, content=body.encode())


@pytest.mark.asyncio
async def test_astream_parses_ndjson_chunks_in_order():
    chunks = [
        {"response": "Hello", "done": False},
        {"response": ", world", "done": False},
        {"response": "!", "done": True, "done_reason": "stop"},
    ]

    def handler(request):
        return _ndjson_response(chunks)

    transport = httpx.MockTransport(handler)
    backend = OllamaBackend(host="http://fake-ollama:11434")

    async with httpx.AsyncClient(transport=transport) as client:
        received = []
        async for chunk in backend.astream("hi", "phi3:mini", 50, client=client):
            received.append(chunk)

    assert received == chunks


@pytest.mark.asyncio
async def test_astream_raises_clear_error_on_model_not_found():
    def handler(request):
        return httpx.Response(404, content=b"model 'phi3:mini' not found")

    transport = httpx.MockTransport(handler)
    backend = OllamaBackend(host="http://fake-ollama:11434")

    with pytest.raises(OllamaConnectionError, match="not found"):
        async with httpx.AsyncClient(transport=transport) as client:
            async for _ in backend.astream("hi", "phi3:mini", 50, client=client):
                pass


@pytest.mark.asyncio
async def test_astream_raises_clear_error_when_ollama_unreachable():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)
    backend = OllamaBackend(host="http://fake-ollama:11434")

    with pytest.raises(OllamaConnectionError, match="couldn't reach Ollama"):
        async with httpx.AsyncClient(transport=transport) as client:
            async for _ in backend.astream("hi", "phi3:mini", 50, client=client):
                pass


@pytest.mark.asyncio
async def test_astream_raises_clear_error_on_generic_http_error():
    """
    Any non-2xx status (not just 404) should surface as a readable
    OllamaConnectionError with the response body visible, not an unhandled
    httpx.HTTPStatusError -- this is what previously left errors like a
    malformed options payload surfacing as an opaque 500 with no detail.
    """

    def handler(request):
        return httpx.Response(500, content=b'{"error":"something went wrong server-side"}')

    transport = httpx.MockTransport(handler)
    backend = OllamaBackend(host="http://fake-ollama:11434")

    with pytest.raises(OllamaConnectionError, match="500"):
        async with httpx.AsyncClient(transport=transport) as client:
            async for _ in backend.astream("hi", "phi3:mini", 50, client=client):
                pass


@pytest.mark.asyncio
async def test_astream_sends_expected_payload():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _ndjson_response([{"response": "ok", "done": True, "done_reason": "stop"}])

    transport = httpx.MockTransport(handler)
    backend = OllamaBackend(host="http://fake-ollama:11434")

    async with httpx.AsyncClient(transport=transport) as client:
        async for _ in backend.astream("explain kv caching", "phi3:mini", 42, client=client):
            pass

    assert captured["body"]["model"] == "phi3:mini"
    assert captured["body"]["prompt"] == "explain kv caching"
    assert captured["body"]["stream"] is True
    assert captured["body"]["options"]["num_predict"] == 42
    assert captured["body"]["options"]["num_ctx"] == 2048
    assert captured["body"]["options"]["temperature"] == 0.7
