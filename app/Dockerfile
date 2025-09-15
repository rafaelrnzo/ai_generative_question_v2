FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_NO_CACHE_DIR=1 \
    # prioritaskan wheel (tanpa compile)
    PIP_PREFER_BINARY=1

WORKDIR /app

# Paksa IPv4 + pakai mirror Indonesia + HTTPS
RUN set -eux; \
    echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4; \
    printf '%s\n' \
      'deb https://kartolo.sby.datautama.net.id/debian bookworm main contrib non-free non-free-firmware' \
      'deb https://kartolo.sby.datautama.net.id/debian bookworm-updates main contrib non-free non-free-firmware' \
      'deb https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware' \
      > /etc/apt/sources.list; \
    apt-get -o Acquire::Retries=5 update; \
    # hanya ca-certificates agar HTTPS pip aman
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends ca-certificates; \
    update-ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# Pip toolchain
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements lebih dulu biar cache build efektif
COPY requirements.txt ./

# 1) Coba install semua via WHEEL saja.
# 2) Jika gagal, pasang build deps MINIMAL (tanpa rust), dan HANYA lib yang kemungkinan dibutuhkan
#    berdasarkan isi requirements.txt, dengan cache APT diarahkan ke /tmp lalu dibersihkan.
RUN set -eux; \
  echo "=== PIP wheel-only attempt ==="; \
  if pip install --only-binary=:all: --prefer-binary -r requirements.txt; then \
    echo ">> Sukses pakai wheel, tidak perlu APT build deps"; \
  else \
    echo ">> Wheel-only gagal, install build deps minimal & retry"; \
    NEED_PG=""; NEED_MYSQL=""; \
    grep -qiE 'psycopg2|(^|[[:space:]])psycopg([[:space:]=<>\[]|$)' requirements.txt && NEED_PG="yes" || true; \
    grep -qi 'mysqlclient' requirements.txt && NEED_MYSQL="yes" || true; \
    # arahkan cache APT ke /tmp agar cepat dibersihkan + jangan simpan paket yang sudah diunduh
    APT_OPTS='-o Acquire::Retries=5 -o Dir::Cache::Archives=/tmp/apt-cache -o APT::Keep-Downloaded-Packages=false'; \
    apt-get ${APT_OPTS} update; \
    # toolchain minimum untuk banyak paket python (tanpa rust)
    apt-get ${APT_OPTS} install -y --no-install-recommends \
      build-essential pkg-config libffi-dev libssl-dev \
      $( [ -n "$NEED_PG" ] && echo libpq-dev || true ) \
      $( [ -n "$NEED_MYSQL" ] && echo default-libmysqlclient-dev || true ); \
    rm -rf /tmp/apt-cache || true; \
    # retry pip install (sekarang boleh compile sedikit)
    pip install -v -r requirements.txt; \
    # bersih-bersih supaya layer tetap kecil
    apt-get purge -y --auto-remove; \
    rm -rf /var/lib/apt/lists/*; \
  fi

# (opsional) paket tambahan
RUN pip install --no-cache-dir openai

# Copy source
COPY . .

EXPOSE 8002
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8002","--reload"]
