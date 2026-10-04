# ============================================================================
# Stage 1: Build ONNX artifacts (torch lives ONLY in this throwaway stage)
#   - Export the MiniLM sentence embedder to ONNX
#   - Convert the trained MLP weights (.pth) to .npz for NumPy inference
# ============================================================================
FROM python:3.12-slim AS onnx-builder

ENV PIP_NO_CACHE_DIR=1

COPY scripts/export_onnx.py /tmp/export_onnx.py
COPY app/classifier_mlp.pth /tmp/classifier_mlp.pth

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "transformers>=4.44,<5" onnx

RUN python /tmp/export_onnx.py /embedder_onnx

# ============================================================================
# Stage 2: Runtime — no PyTorch, no build tools
# ============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Step 1: Install runtime dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Step 2: Drop in the pre-built ONNX embedder + MLP weights
COPY --from=onnx-builder /embedder_onnx /app/app/onnx_embedder
COPY --from=onnx-builder /mlp/classifier_mlp.npz /app/app/classifier_mlp.npz

# Step 3: Copy source
COPY . .

# Step 4: Run as a non-root user
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',timeout=4).status==200 else 1)"

# Bind to $PORT (Render/Cloud Run inject it); defaults to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
