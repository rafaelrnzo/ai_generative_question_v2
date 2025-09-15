# Pakai base yang stabil + kecil
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Paksa APT pakai IPv4 + pakai HTTPS di semua sumber
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
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    ; \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --upgrade pip setuptools wheel

# Install deps Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# (Kalau openai sudah ada di requirements.txt, bagian ini tak perlu)
# RUN pip install --no-cache-dir openai

# Copy source code
COPY . .

EXPOSE 8002

# Note: --reload untuk dev; hapus di production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
