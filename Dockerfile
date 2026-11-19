# The prediction API, with the champion trained inside the build from the pinned data.
#
#   docker build -t telco-churn .
#   docker run --rm -p 8000:8000 telco-churn
#   open http://localhost:8000/docs
ARG BASE_IMAGE=python:3.12-slim

# ---- Stage 1: install the locked dependencies and train the model -----------------
FROM ${BASE_IMAGE} AS build

RUN pip install --no-cache-dir uv==0.11.32

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

# Dependencies first, so a change under src/ doesn't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra serve --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev --extra serve --no-editable

# Download the dataset (checked against its SHA-256) and train the champion.
# No TensorFlow here: the served model is the scikit-learn champion.
RUN .venv/bin/churn train --no-nn

# ---- Stage 2: only what the API needs to run ---------------------------------------
FROM ${BASE_IMAGE}

RUN useradd --create-home api
WORKDIR /app
COPY --from=build --chown=api:api /app/.venv ./.venv
COPY --from=build --chown=api:api /app/models ./models

USER api
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
