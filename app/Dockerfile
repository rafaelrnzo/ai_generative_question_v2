# ✅ Stable + small base
FROM python:3.11-slim-bookworm

# --- Env & pip defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Optional: quieter noninteractive apt
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ✅ Force IPv4, ensure HTTPS mirrors exist for both styles, then install only essentials
RUN set -eux; \
    echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4; \
    # Switch http->https in classic sources.list if present
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list; \
        sed -i 's|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list; \
    fi; \
    # Switch http->https in debian.sources (bookworm uses this)
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get -o Acquire::ForceIPv4=true update; \
    apt-get -o Acquire::ForceIPv4=true install -y --no-install-recommends \
        ca-certificates \
        build-essential \
    ; \
    rm -rf /var/lib/apt/lists/*

# ✅ Upgrade pip tooling (kept compact)
RUN python -m pip install --upgrade pip setuptools wheel

# ✅ Dependency layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# ✅ App code
COPY . .

# (Optional) run as non-root for safety:
# RUN useradd -m appuser && chown -R appuser:appuser /app
# USER appuser

EXPOSE 8002

# ⚠️ --reload for dev only. Remove for production.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
