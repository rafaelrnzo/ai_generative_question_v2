# Use a specific version for reproducibility
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Update package lists and install build-essential and other dependencies in one step
# This reduces layer size and ensures a clean state
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    # Clean up the apt cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Install/upgrade Python packaging tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Note: If 'openai' is in your requirements.txt, the next line is redundant.
# If not, you can keep it. For this example, I'll assume it's in the requirements file.
# RUN pip install --no-cache-dir openai

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8002

# Define the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]