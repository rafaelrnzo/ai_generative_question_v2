FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Copy deps first for caching
COPY requirements.txt .

# Install deps (prefer wheels)
RUN python -m pip install --prefer-binary -r requirements.txt

# If openai isn’t in requirements.txt, uncomment:
# RUN python -m pip install --prefer-binary openai

# Copy the app
COPY . .

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
