FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

# Avoid IPv6 apt hiccups seen in your first error
RUN printf 'Acquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99force-ipv4

# System deps commonly needed for building Python packages + git for VCS requirements
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
      libxslt1-dev \
      libjpeg-dev \
      zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for layering
COPY requirements.txt .

# Prefer prebuilt wheels to dodge native builds; add -v for clear error logs
# If you know you need to build from source, remove --prefer-binary
RUN python -m pip install --prefer-binary -v -r requirements.txt

# If openai is not in requirements.txt, uncomment:
# RUN python -m pip install --prefer-binary -v openai

# Copy the rest of the app
COPY . .

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
