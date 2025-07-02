# cloudrun/Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir google-cloud-storage feedparser Flask gunicorn openai
COPY . .
CMD ["bash", "-c", "gunicorn --bind 0.0.0.0:$PORT news_summarizer_runner:app"]