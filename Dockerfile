FROM python:3.11-slim

WORKDIR /app

# system deps for numpy wheels are already covered by manylinux wheels on
# python:slim, so no extra apt packages are needed here
COPY pyproject.toml ./
COPY mini_vllm ./mini_vllm

RUN pip install --no-cache-dir .

EXPOSE 8000

ENV MINI_VLLM_NUM_BLOCKS=2048 \
    MINI_VLLM_BLOCK_SIZE=16 \
    MINI_VLLM_MAX_NUM_SEQS=32 \
    MINI_VLLM_MAX_BATCHED_TOKENS=4096 \
    MINI_VLLM_PREFILL_CHUNK=512

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "mini_vllm.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
