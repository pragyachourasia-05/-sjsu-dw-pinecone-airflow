FROM apache/airflow:2.8.1

USER root

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    sentence-transformers==3.1.1 \
    pinecone==5.3.1