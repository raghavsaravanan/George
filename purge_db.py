#!/usr/bin/env python3
"""Wipe George's Qdrant collection for a clean from-scratch re-ingest.

Connects to Qdrant Cloud via QDRANT_URL + QDRANT_API_KEY (from .env / env).
Drops the active george_specs collection and recreates it empty with the
same vector size and keyword indexes used by ingest.py.

Usage:
  .venv/bin/python purge_db.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

COLLECTION_NAME = "george_specs"
VECTOR_SIZE = 384
INDEX_FIELDS = ("document_ref", "year", "make", "model", "shop_id")
DEFAULT_SHOP_ID = "shop_demo"


def open_cloud_client() -> QdrantClient:
    qdrant_url = (os.getenv("QDRANT_URL") or "").strip()
    qdrant_api_key = (os.getenv("QDRANT_API_KEY") or "").strip()
    if not qdrant_url or not qdrant_api_key:
        raise SystemExit(
            "QDRANT_URL and QDRANT_API_KEY must be set (e.g. in .env) "
            "before running purge_db.py."
        )
    print(f"Connecting to Qdrant Cloud: {qdrant_url}")
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30.0)


def recreate_empty_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        before = getattr(info, "points_count", None)
        print(
            f"Found collection {COLLECTION_NAME!r} "
            f"(points_count={before}). Dropping..."
        )
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped collection {COLLECTION_NAME!r}.")
    else:
        print(f"Collection {COLLECTION_NAME!r} did not exist.")

    print(f"Creating empty collection {COLLECTION_NAME!r} (dim={VECTOR_SIZE})...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=VECTOR_SIZE,
            distance=qmodels.Distance.COSINE,
        ),
    )

    for field_name in INDEX_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        print(f"  payload index ready: {field_name}")


def verify_empty(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        raise SystemExit(f"ERROR: collection {COLLECTION_NAME!r} missing after recreate.")

    info = client.get_collection(COLLECTION_NAME)
    count = int(getattr(info, "points_count", 0) or 0)

    # Also confirm the demo tenant has zero points (filter query).
    demo_hits = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="shop_id",
                    match=qmodels.MatchValue(value=DEFAULT_SHOP_ID),
                )
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    demo_points = list(demo_hits[0]) if demo_hits else []

    print("---")
    print(f"Collection {COLLECTION_NAME!r} points_count={count}")
    print(f"shop_id={DEFAULT_SHOP_ID!r} visible points={len(demo_points)}")
    if count != 0 or demo_points:
        raise SystemExit("ERROR: database is NOT empty after purge.")
    print("DATABASE OFFICIALLY EMPTY. Safe to re-ingest with ingest.py.")


def main() -> None:
    client = open_cloud_client()
    try:
        recreate_empty_collection(client)
        verify_empty(client)
    finally:
        client.close()
        print("Connection closed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Purge interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
