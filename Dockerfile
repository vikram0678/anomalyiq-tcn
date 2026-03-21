FROM python:3.10-slim



WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    torch==2.2.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    numpy==1.24.0 \
    pandas==2.0.0 \
    scikit-learn==1.3.0 \
    streamlit==1.28.0 \
    plotly==5.17.0 \
    scipy==1.11.0 \
    requests==2.31.0 \
    joblib==1.3.0 \
    matplotlib==3.7.0

COPY . .

RUN mkdir -p data/raw data/processed models results

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]