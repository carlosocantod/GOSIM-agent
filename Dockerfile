FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install uv

ENV PYTHONPATH=/app

# copy only dependency files first (for caching)
COPY pyproject.toml uv.lock* ./

# install deps
RUN uv sync --no-dev

# now copy code
COPY src/ ./src/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
