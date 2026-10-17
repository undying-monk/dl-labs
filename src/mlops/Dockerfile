FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Compile Python source files to bytecode for faster startup times
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1

# Work from the workspace root context
WORKDIR /workspace

# Step 1: Copy only package configurations to cache dependency installation layers
COPY pyproject.toml uv.lock ./
COPY src/mlops/pyproject.toml ./apps/mlops/

# Step 2: Sync dependencies without source code (creates the layout)
RUN uv sync --frozen --no-dev --no-install-workspace --package mlops

# Step 3: Copy the required source code 
COPY src/mlops ./apps/mlops/src

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


WORKDIR /app

# Copy configuration files first to optimize Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies exactly as defined in the lockfile
RUN uv sync --frozen --no-cache --no-dev

# Copy the rest of your application code
COPY . .
COPY model/ ./model/

ENV MODEL_URI=/app/model

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "iris_service.app:app", "--host", "0.0.0.0", "--port", "8000"]