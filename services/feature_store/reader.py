"""
reader.py — Rant-Free Feature Store
Fungsi read dari MongoDB untuk Training Service dan Monitoring Service.

Sembunyikan kompleksitas aggregation pipeline MongoDB di sini —
service lain cukup import fungsi yang dibutuhkan tanpa perlu tahu
detail implementasi penyimpanan Feast.
"""

import logging
import os
from typing import Optional

import pandas as pd
from pymongo import MongoClient

log = logging.getLogger(__name__)


def _get_client() -> tuple[MongoClient, str, str]:
    """Return (client, database, collection) dari env var."""
    conn = os.environ.get("MONGODB_CONNECTION_STRING")
    if not conn:
        raise RuntimeError(
            "MONGODB_CONNECTION_STRING tidak di-set — "
            "read dari MongoDB tidak tersedia di mode local dev."
        )
    database = os.environ.get("MONGODB_DATABASE", "feast")
    collection = os.environ.get("MONGODB_COLLECTION", "feature_history")
    return MongoClient(conn), database, collection


def read_training_data(min_event_timestamp: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Baca semua data yang punya label untuk Training Service.

    Mengembalikan satu baris per request_id (baris terbaru — verification row
    menang atas prediction row). Baris tanpa label (toxic=null) dibuang.

    Args:
        min_event_timestamp: Jika di-set, hanya ambil baris dengan
            event_timestamp >= nilai ini. Berguna untuk replay buffer
            (pisahkan data "baru" dari data historis).

    Returns:
        DataFrame dengan kolom: text, score_toxic, confidence,
        model_version, toxic, event_timestamp.
    """
    client, database, collection = _get_client()
    try:
        coll = client[database][collection]

        match_stage: dict = {"feature_view": "comment_features"}
        if min_event_timestamp is not None:
            match_stage["event_timestamp"] = {"$gte": min_event_timestamp.to_pydatetime()}

        pipeline = [
            {"$match": match_stage},
            {"$sort": {"created_at": -1}},
            {"$group": {
                "_id": "$entity_id",
                "features": {"$first": "$features"},
                "event_timestamp": {"$first": "$event_timestamp"},
            }},
            {"$match": {"features.toxic": {"$ne": None}}},
        ]

        docs = list(coll.aggregate(pipeline))
        log.info(f"read_training_data: {len(docs):,} baris ditemukan")

        rows = []
        for doc in docs:
            f = doc["features"]
            rows.append({
                "event_timestamp": doc["event_timestamp"],
                "text":          f.get("text"),
                "score_toxic":   f.get("score_toxic"),
                "confidence":    f.get("confidence"),
                "model_version": f.get("model_version"),
                "toxic":         f.get("toxic"),
            })

        return pd.DataFrame(rows)

    finally:
        client.close()


def read_monitoring_data(min_event_timestamp: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Baca semua prediction rows untuk Monitoring Service (drift + fairness).

    Berbeda dengan read_training_data — di sini kita butuh prediction rows
    (score_toxic tidak null) terlepas dari ada tidaknya label verifikasi,
    karena monitoring menganalisis distribusi prediksi model dari waktu ke waktu.

    Args:
        min_event_timestamp: Jika di-set, hanya ambil baris dengan
            event_timestamp >= nilai ini. Gunakan ini untuk membatasi
            window monitoring (misal: 7 hari terakhir).

    Returns:
        DataFrame dengan kolom: text, score_toxic, confidence,
        model_version, toxic, event_timestamp.
    """
    client, database, collection = _get_client()
    try:
        coll = client[database][collection]

        match_stage: dict = {
            "feature_view": "comment_features",
            "features.score_toxic": {"$ne": None},  # hanya prediction rows
        }
        if min_event_timestamp is not None:
            match_stage["event_timestamp"] = {"$gte": min_event_timestamp.to_pydatetime()}

        pipeline = [
            {"$match": match_stage},
            {"$sort": {"created_at": -1}},
            {"$group": {
                "_id": "$entity_id",
                "features": {"$first": "$features"},
                "event_timestamp": {"$first": "$event_timestamp"},
            }},
        ]

        docs = list(coll.aggregate(pipeline))
        log.info(f"read_monitoring_data: {len(docs):,} baris ditemukan")

        rows = []
        for doc in docs:
            f = doc["features"]
            rows.append({
                "event_timestamp": doc["event_timestamp"],
                "text":          f.get("text"),
                "score_toxic":   f.get("score_toxic"),
                "confidence":    f.get("confidence"),
                "model_version": f.get("model_version"),
                "toxic":         f.get("toxic"),
            })

        return pd.DataFrame(rows)

    finally:
        client.close()
