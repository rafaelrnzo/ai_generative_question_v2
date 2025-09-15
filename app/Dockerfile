FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN set -eux; \
    echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4; \
    printf '%s\n' \
      'deb https://kartolo.sby.datautama.net.id/debian bookworm main contrib non-free non-free-firmware' \
      'deb https://kartolo.sby.datautama.net.id/debian bookworm-updates main contrib non-free non-free-firmware' \
      'deb https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware' \
      > /etc/apt/sources.list; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
      ca-certificates; \
    update-ca-certificates; \
    # kalau perlu toolchain:
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
      build-essential; \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# opsional:
RUN pip install --no-cache-dir openai

COPY . .
EXPOSE 8002
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8002","--reload"]
