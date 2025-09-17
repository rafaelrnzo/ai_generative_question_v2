FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Force APT to use IPv4 to avoid IPv6 routing hiccups
RUN printf 'Acquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99force-ipv4

# (Usually you don't need build-essential for pure-Python/wheel deps)
WORKDIR /app

# Upgrade pip tooling first
RUN pip install --upgrade pip setuptools wheel

# Copy and install deps (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# If openai isn't already in requirements.txt, keep this; otherwise delete it
# RUN pip install --no-cache-dir openai

# Copy the rest of your app
COPY . .

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
