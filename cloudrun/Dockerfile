# cloudrun/Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir google-cloud-storage feedparser
COPY fetch_news_runner.py .
CMD ["python", "fetch_news_runner.py"]