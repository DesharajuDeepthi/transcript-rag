FROM python:3.11-slim

WORKDIR /app

# System deps: build-essential for chromadb native libs, curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "mcp>=1.0.0"

# Copy source — data/ and outputs/ are mounted as volumes at runtime
COPY src/ ./src/
COPY mcp_server/ ./mcp_server/
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/
COPY scripts/ ./scripts/

# Install the project itself as an editable package (exposes src + mcp_server)
RUN pip install --no-cache-dir -e .

# Default: open a bash shell; override at docker-compose level
CMD ["bash"]
