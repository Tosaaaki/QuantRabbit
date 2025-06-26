# QuantRabbit/cloudrun/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "news_summarizer_runner.py"]   # ← build 時に上書き可