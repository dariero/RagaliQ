# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: builder
# Builds the wheel inside a dedicated layer so build tools don't end up
# in the final image. Only the compiled .whl is passed to the runtime stage.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Copy only the files needed to build the package.
# Separating COPY steps this way means Docker can cache the pip install layer
# independently of source code changes.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /dist

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime
# Starts from a clean slim base — no build tools, no source, no cache.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="RagaliQ" \
      org.opencontainers.image.description="LLM & RAG Evaluation Testing Framework" \
      org.opencontainers.image.source="https://github.com/dariero/RagaliQ" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Copy the wheel from the builder stage and install it.
# --no-cache-dir keeps the image smaller by not storing pip's download cache.
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/ragaliq-*.whl \
    && rm /tmp/ragaliq-*.whl

# The ANTHROPIC_API_KEY must be supplied at runtime:
#   docker run -e ANTHROPIC_API_KEY=sk-ant-... ghcr.io/dariero/ragaliq run ...
# Never bake secrets into the image.

ENTRYPOINT ["ragaliq"]
CMD ["--help"]
