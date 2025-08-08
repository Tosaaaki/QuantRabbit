FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir google-cloud-storage feedparser Flask gunicorn openai
COPY . .
CMD ["sh", "-c", "gunicorn --workers=1 --bind 0.0.0.0:$PORT news_summarizer_runner:app"]