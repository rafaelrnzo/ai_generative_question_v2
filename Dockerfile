FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

# Avoid IPv6 issues that previously broke apt
RUN printf 'Acquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99force-ipv4

# System build deps commonly needed by Python packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      git \
      pkg-config \
      libffi-dev \
      libssl-dev \
      libpq-dev \
      libxml2-dev \
      libxslt-dev \
      libjpeg62-turbo-dev \
      zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install deps (prefer wheels; verbose to see exact failing pkg if any)
RUN python -m pip install --prefer-binary -v -r requirements.txt

# If openai isn't in requirements.txt, uncomment:
# RUN python -m pip install --prefer-binary -v openai

# Copy the rest
COPY . .

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
