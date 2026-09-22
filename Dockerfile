# syntax=docker/dockerfile:1.4
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Step 1: Copy only requirements first (allows layer caching)
COPY requirements.txt .

# Step 2: Install dependencies with persistent pip cache mount
# (If interrupted or requirements change slightly, downloaded wheels are reused)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Step 3: Pre-download SentenceTransformer model with persistent HuggingFace cache
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Step 4: Copy application source code (put last because code changes most often)
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI application with standard production flags
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]