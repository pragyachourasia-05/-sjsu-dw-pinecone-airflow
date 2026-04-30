from __future__ import annotations

import json
import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

DATA_DIR   = "/opt/airflow/data"
INPUT_FILE = os.path.join(DATA_DIR, "pinecone_input.json")
INDEX_NAME = "sjsu-dw-index"
DIMENSION  = 384
METRIC     = "cosine"
SEARCH_QUERY = "What are the key concepts of data warehousing?"


def download_and_process(**context):
    os.makedirs(DATA_DIR, exist_ok=True)

    documents = [
        {"id": "doc_001", "text": "A data warehouse is a centralized repository for structured data from multiple sources.", "category": "definition"},
        {"id": "doc_002", "text": "ETL stands for Extract, Transform, Load — the core process of moving data into a warehouse.", "category": "etl"},
        {"id": "doc_003", "text": "OLAP enables complex queries and multidimensional analysis on large datasets.", "category": "olap"},
        {"id": "doc_004", "text": "Star schema organizes fact tables surrounded by dimension tables for fast query performance.", "category": "schema"},
        {"id": "doc_005", "text": "Snowflake schema normalizes dimension tables into multiple related tables.", "category": "schema"},
        {"id": "doc_006", "text": "Apache Airflow is a platform to programmatically author, schedule, and monitor workflows.", "category": "tools"},
        {"id": "doc_007", "text": "Pinecone is a managed vector database designed for similarity search at scale.", "category": "tools"},
        {"id": "doc_008", "text": "Sentence transformers convert sentences into dense vector embeddings for semantic search.", "category": "ml"},
        {"id": "doc_009", "text": "Data marts are subsets of a data warehouse focused on specific business areas.", "category": "definition"},
        {"id": "doc_010", "text": "Slowly Changing Dimensions track historical changes in dimension data over time.", "category": "concept"},
        {"id": "doc_011", "text": "Columnar storage formats like Parquet improve analytical query performance significantly.", "category": "storage"},
        {"id": "doc_012", "text": "Data lineage tracks the origin and transformations applied to data throughout its lifecycle.", "category": "governance"},
        {"id": "doc_013", "text": "Real-time data streaming with Kafka enables low-latency ingestion into modern data pipelines.", "category": "streaming"},
        {"id": "doc_014", "text": "Dimensional modeling is a design technique optimizing databases for data warehouse queries.", "category": "modeling"},
        {"id": "doc_015", "text": "Semantic search uses meaning and context rather than exact keyword matching to find results.", "category": "ml"},
    ]

    with open(INPUT_FILE, "w") as f:
        json.dump(documents, f, indent=2)

    logging.info(f"Written {len(documents)} documents to {INPUT_FILE}")
    context["ti"].xcom_push(key="doc_count", value=len(documents))


def create_pinecone_index(**context):
    from pinecone import Pinecone, ServerlessSpec
    import time

    api_key = Variable.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing:
        logging.info(f"Index '{INDEX_NAME}' already exists — skipping creation.")
    else:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logging.info(f"Created index '{INDEX_NAME}' (dim={DIMENSION}, metric={METRIC})")

    while not pc.describe_index(INDEX_NAME).status["ready"]:
        logging.info("Waiting for index to become ready...")
        time.sleep(5)

    logging.info(f"Index '{INDEX_NAME}' is ready.")


def embed_and_upsert(**context):
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone

    api_key = Variable.get("PINECONE_API_KEY")
    pc    = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    with open(INPUT_FILE) as f:
        documents = json.load(f)

    logging.info(f"Loaded {len(documents)} documents from {INPUT_FILE}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [doc["text"] for doc in documents]
    logging.info("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=False)
    logging.info(f"Generated {len(embeddings)} embeddings of dim {embeddings.shape[1]}")

    vectors = [
        {
            "id": doc["id"],
            "values": emb.tolist(),
            "metadata": {"text": doc["text"], "category": doc["category"]},
        }
        for doc, emb in zip(documents, embeddings)
    ]

    batch_size = 50
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        logging.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

    stats = index.describe_index_stats()
    logging.info(f"Upsert complete. Index stats: {stats}")


def search_pinecone(**context):
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone

    api_key = Variable.get("PINECONE_API_KEY")
    pc    = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    model     = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([SEARCH_QUERY])[0].tolist()

    logging.info(f"Searching for: '{SEARCH_QUERY}'")
    results = index.query(
        vector=query_vec,
        top_k=5,
        include_metadata=True,
    )

    logging.info("Top-5 Results:")
    for i, match in enumerate(results["matches"], 1):
        score = round(match["score"], 4)
        text  = match["metadata"]["text"]
        cat   = match["metadata"]["category"]
        logging.info(f"  {i}. [{score}] ({cat}) {text}")

    output_path = os.path.join(DATA_DIR, "search_results.json")
    with open(output_path, "w") as f:
        json.dump(
            {
                "query": SEARCH_QUERY,
                "results": [
                    {
                        "rank": i + 1,
                        "score": m["score"],
                        "id": m["id"],
                        "text": m["metadata"]["text"],
                        "category": m["metadata"]["category"],
                    }
                    for i, m in enumerate(results["matches"])
                ],
            },
            f,
            indent=2,
        )
    logging.info(f"Results saved to {output_path}")


with DAG(
    dag_id="pinecone_pipeline",
    default_args=default_args,
    description="Download -> Process -> Create Index -> Embed & Upsert -> Search",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["pinecone", "embeddings", "data-warehouse"],
) as dag:

    t1_download = PythonOperator(
        task_id="download_and_process_data",
        python_callable=download_and_process,
    )

    t2_create_index = PythonOperator(
        task_id="create_pinecone_index",
        python_callable=create_pinecone_index,
    )

    t3_embed_upsert = PythonOperator(
        task_id="embed_and_upsert_to_pinecone",
        python_callable=embed_and_upsert,
    )

    t4_search = PythonOperator(
        task_id="search_pinecone",
        python_callable=search_pinecone,
    )

    t1_download >> t2_create_index >> t3_embed_upsert >> t4_search
