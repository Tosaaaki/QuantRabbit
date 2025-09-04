FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY cloudrun/requirements.txt cloudrun/requirements.txt
RUN pip install --no-cache-dir -r cloudrun/requirements.txt && pip install gunicorn
COPY . .
# Use the GUNICORN_APP env to select which service to run: cloudrun.news_summarizer_runner:app, cloudrun.fetch_news_runner:app,
# cloudrun.news_ingestor_service:app, cloudrun.trader_service:app. Default to summarizer for backwards compatibility.
ENV GUNICORN_APP=cloudrun.news_summarizer_runner:app
CMD ["sh", "-c", "gunicorn --workers=1 --bind 0.0.0.0:$PORT ${GUNICORN_APP}"]
