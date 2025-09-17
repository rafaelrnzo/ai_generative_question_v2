FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# This part remains the same to cache the dependencies
COPY requirements.txt .
RUN python -m pip install --prefer-binary -r requirements.txt

# --- THIS IS THE MODIFIED LINE ---
# Instead of copying the whole project, copy only the contents
# of the 'app' directory into the container's /app directory.
COPY app/ .

EXPOSE 8002

# Now this command will work perfectly because main.py is in the root /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]