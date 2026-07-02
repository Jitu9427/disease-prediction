# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Set working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Install system dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Copy and install Python dependencies ──────────────────────────────────────
COPY app-requirements.txt .
RUN pip install --no-cache-dir -r app-requirements.txt

# ── Copy the Flask application ─────────────────────────────────────────────────
COPY app/ .

# ── Expose the port Flask runs on ─────────────────────────────────────────────
EXPOSE 5000

# ── Run the app with gunicorn (production-ready server) ───────────────────────
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
