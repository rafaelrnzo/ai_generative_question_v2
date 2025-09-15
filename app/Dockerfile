# ---------- Base ----------
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1 \
    # Prefer wheel (tanpa compile) sebisa mungkin
    PIP_PREFER_BINARY=1

WORKDIR /app

# ---------- APT hardening + mirror Indonesia ----------
RUN set -eux; \
    echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4; \
    printf '%s\n' \
      'deb https://kartolo.sby.datautama.net.id/debian bookworm main contrib non-free non-free-firmware' \
      'deb https://kartolo.sby.datautama.net.id/debian bookworm-updates main contrib non-free non-free-firmware' \
      'deb https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware' \
      > /etc/apt/sources.list; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends ca-certificates; \
    update-ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# ---------- pip tooling ----------
RUN python -m pip install --upgrade pip setuptools wheel

# ---------- install Python deps (2 tahap: wheel-only -> fallback compile) ----------
COPY requirements.txt ./

# 1) Coba install dengan wheel saja (verbose supaya kelihatan paket yang gagal)
# 2) Kalau gagal, install toolchain + libs umum lalu retry pip install
RUN set -eux; \
    echo "=== PIP (wheel-only) attempt ==="; \
    if ! pip install -v -r requirements.txt; then \
      echo "=== Wheel-only gagal. Pasang build deps & retry ==="; \
      apt-get -o Acquire::Retries=5 update; \
      apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        default-libmysqlclient-dev \
        rustc cargo \
      ; \
      pip install -v -r requirements.txt; \
      # Bersih-bersih layer APT
      apt-get purge -y --auto-remove; \
      rm -rf /var/lib/apt/lists/*; \
    fi

# (opsional) paket tambahan lain
RUN pip install --no-cache-dir openai

# ---------- app ----------
COPY . .

EXPOSE 8002
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8002","--reload"]
