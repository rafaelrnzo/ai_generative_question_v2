# Base kecil & stabil
FROM python:3.11-slim-bookworm

# Non-interaktif & Python nyaman di container
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Hardening APT: paksa IPv4 + pakai HTTPS di semua sumber ---
# Catatan:
# - Debian modern kadang pakai /etc/apt/sources.list.d/debian.sources (format baru)
# - Kita patch KEDUANYA kalau ada
RUN set -eux; \
    echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4; \
    # Pastikan HTTPS di sources.list (kalau file-nya ada)
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list; \
        sed -i 's|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list; \
    fi; \
    # Pastikan HTTPS di format debian.sources (kalau file-nya ada)
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    # Update dengan retry + IPv4, install cert dulu supaya HTTPS jalan
    apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 update; \
    apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 install -y --no-install-recommends \
        ca-certificates \
        apt-transport-https \
    ; \
    update-ca-certificates

# --- Build deps minimal (gcc/make) ---
RUN set -eux; \
    apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 update; \
    apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=3 install -y --no-install-recommends \
        build-essential \
    ; \
    rm -rf /var/lib/apt/lists/*

# --- Pip tooling terbaru, non-cache untuk layer lebih ramping ---
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements dulu biar cache build optimal
COPY requirements.txt ./

# Gunakan timeout + no-cache-dir; kalau repository Python via HTTPS saja
RUN pip install --default-timeout=120 --no-cache-dir -r requirements.txt

# (opsional) paket tambahan yang kamu sebut
RUN pip install --no-cache-dir openai

# Sisanya
COPY . .

EXPOSE 8002

# Untuk dev kamu memang pakai --reload; aman kalau di dev
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
