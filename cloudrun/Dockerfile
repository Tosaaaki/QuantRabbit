FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir flask openai google-cloud-storage
COPY news_summarizer_runner.py .
CMD ["python", "news_summarizer_runner.py"]