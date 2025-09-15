# ✅ Stable + kecil base image
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ✅ Force IPv4 & pakai HTTPS di apt
RUN set -eux; \
    echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list; \
        sed -i 's|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list; \
    fi; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get -o Acquire::ForceIPv4=true update; \
    apt-get -o Acquire::ForceIPv4=true install -y --no-install-recommends \
        ca-certificates \
        build-essential \
    ; \
    rm -rf /var/lib/apt/lists/*

# ✅ Upgrade pip tools
RUN python -m pip install --upgrade pip setuptools wheel

# ✅ Copy & install deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# ✅ Copy source code (langsung ke /app)
COPY . .

EXPOSE 8002

# ⚠️ Dev mode pakai --reload, di prod ganti --workers
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
