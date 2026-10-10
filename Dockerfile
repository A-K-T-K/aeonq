FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE requirements-lock.txt ./
# Pinned third-party packages first, so source edits reuse this cached layer.
RUN grep -v '^-e ' requirements-lock.txt > /tmp/third-party-lock.txt \
    && python -m pip install --no-cache-dir -r /tmp/third-party-lock.txt
COPY aeonq ./aeonq
RUN python -m pip install --no-cache-dir --no-deps -e .
COPY tests ./tests
COPY benchmarks ./benchmarks

CMD ["python", "-m", "pytest", "-q"]
