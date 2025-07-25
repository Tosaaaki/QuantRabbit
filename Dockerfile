FROM python:3.11-slim

ARG ENV_PATH=config/env.example.toml

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential curl
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ${ENV_PATH} ./config/env.toml

CMD ["python", "main.py"]