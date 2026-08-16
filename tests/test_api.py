import concurrent.futures

from fastapi.testclient import TestClient

from mini_vllm.api.app import app


def test_healthz():
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_generate_with_text_prompt():
    with TestClient(app) as client:
        resp = client.post("/v1/generate", json={"prompt": "hello there world", "max_tokens": 4})
        assert resp.status_code == 200
        data = resp.json()
        assert data["tokens_generated"] == 4
        assert data["finish_reason"] == "length"
        assert len(data["output_token_ids"]) == 4
        assert data["total_latency_ms"] >= 0


def test_generate_with_explicit_token_ids():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/generate", json={"prompt_token_ids": [1, 2, 3, 4, 5], "max_tokens": 2}
        )
        assert resp.status_code == 200
        assert resp.json()["prompt_len"] == 5


def test_generate_missing_prompt_returns_422():
    with TestClient(app) as client:
        resp = client.post("/v1/generate", json={"max_tokens": 2})
        assert resp.status_code == 422


def test_request_status_endpoint():
    with TestClient(app) as client:
        resp = client.post("/v1/generate", json={"prompt": "status check", "max_tokens": 2})
        rid = resp.json()["request_id"]
        status_resp = client.get(f"/v1/requests/{rid}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "finished"


def test_request_status_unknown_id_returns_404():
    with TestClient(app) as client:
        resp = client.get("/v1/requests/does-not-exist")
        assert resp.status_code == 404


def test_stats_endpoint():
    with TestClient(app) as client:
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "cache" in data
        assert "num_running" in data


def test_generate_real_with_faked_backend():
    """
    Exercises the full /v1/generate/real request path without needing an
    actual Ollama server -- swaps in a fake backend at the same seam the
    real one plugs into.
    """
    with TestClient(app) as client:
        from mini_vllm.api import app as app_module

        class FakeBackend:
            async def astream(self, prompt, model, max_tokens, client=None):
                for i, piece in enumerate(["Paged", "Attention", " works"]):
                    yield {
                        "response": piece,
                        "done": i == 2,
                        "done_reason": "stop" if i == 2 else None,
                    }

        app_module._real_engine.backend = FakeBackend()

        resp = client.post(
            "/v1/generate/real", json={"prompt": "what is kv caching", "max_tokens": 20}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["generated_text"] == "PagedAttention works"
        assert data["finish_reason"] == "stop"
        assert data["tokens_generated"] == 3


def test_generate_real_returns_503_when_ollama_unreachable():
    with TestClient(app) as client:
        from mini_vllm.api import app as app_module
        from mini_vllm.backends.ollama_backend import OllamaConnectionError

        class FailingBackend:
            async def astream(self, prompt, model, max_tokens, client=None):
                raise OllamaConnectionError("couldn't reach Ollama at http://fake -- is `ollama serve` running?")
                yield  # pragma: no cover - unreachable, satisfies generator syntax

        app_module._real_engine.backend = FailingBackend()

        resp = client.post("/v1/generate/real", json={"prompt": "hi", "max_tokens": 5})
        assert resp.status_code == 503
        assert "ollama serve" in resp.json()["detail"]


def test_generate_real_stats_endpoint():
    with TestClient(app) as client:
        resp = client.get("/v1/generate/real/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "max_num_seqs" in data


def test_generate_text_format_returns_plain_text_with_real_newlines():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/generate?format=text", json={"prompt": "hello world", "max_tokens": 3}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert "finish_reason=" in resp.text
        assert "\\n" not in resp.text  # no literal backslash-n, real newline instead
        assert "\n" in resp.text


def test_generate_real_text_format_with_faked_backend():
    with TestClient(app) as client:
        from mini_vllm.api import app as app_module

        class FakeBackend:
            async def astream(self, prompt, model, max_tokens, client=None):
                for i, piece in enumerate(["Line one.\n", "Line two."]):
                    yield {
                        "response": piece,
                        "done": i == 1,
                        "done_reason": "stop" if i == 1 else None,
                    }

        app_module._real_engine.backend = FakeBackend()

        resp = client.post(
            "/v1/generate/real?format=text", json={"prompt": "hi", "max_tokens": 20}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert "Line one." in resp.text
        assert "Line two." in resp.text
        assert "finish_reason=stop" in resp.text
        assert "\\n" not in resp.text.replace("Line one.", "")  # only the real newline we sent


def test_generate_real_stream_done_event_includes_finish_reason():
    with TestClient(app) as client:
        from mini_vllm.api import app as app_module

        class FakeBackend:
            async def astream(self, prompt, model, max_tokens, client=None):
                for i, piece in enumerate(["Hello", " world"]):
                    yield {
                        "response": piece,
                        "done": i == 1,
                        "done_reason": "stop" if i == 1 else None,
                    }

        app_module._real_engine.backend = FakeBackend()

        with client.stream(
            "POST", "/v1/generate/real/stream", json={"prompt": "hi", "max_tokens": 20}
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())

        assert "event: token" in body
        assert "event: done" in body
        assert '"finish_reason": "stop"' in body
        assert '"tokens_generated": 2' in body


def test_concurrent_requests_are_all_served_correctly():
    """
    Fire several /v1/generate calls concurrently and confirm every one comes
    back with the token count it asked for -- this is what actually proves
    they were multiplexed through the shared continuous-batching loop rather
    than queued and run one-by-one.
    """
    with TestClient(app) as client:

        def call(i: int):
            return client.post(
                "/v1/generate", json={"prompt": f"request number {i}", "max_tokens": 3 + i % 3}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(call, range(8)))

        for i, resp in enumerate(results):
            assert resp.status_code == 200
            data = resp.json()
            assert data["tokens_generated"] == 3 + i % 3