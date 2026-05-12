"""
Feature Definitions — Rant-Free Feature Store
Mengikuti skema di: IF5251 Laporan Tugas Kelompok - Rant-Free.pdf

Entity  : request_id (bukan id!)
Features: text, score_toxic, confidence, model_version, toxic
          — tidak ada field lain (text_length, word_count, dll.)
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.infra.offline_stores.contrib.mongodb_offline_store.mongodb import MongoDBSource
import os
from feast.types import Float32, Int32, String
from feast.value_type import ValueType

# ── Entity ───────────────────────────────────────────────────────────────────
# Join key adalah request_id — UUID yang di-generate Model Service per request.
comment = Entity(
    name="request_id",
    value_type=ValueType.STRING,
    description="UUID yang di-generate oleh Model Service untuk setiap request prediksi",
)

# ── Offline Source ────────────────────────────────────────────────────────────
# Path relatif dari repo_path (jangan pakai path absolut / Windows)
comment_source_local = FileSource(
    path="data/dataset.parquet",
    timestamp_field="event_timestamp",
)

comment_source_mongo = MongoDBSource(
    name="comment_features",
    timestamp_field="event_timestamp"
)

# Pilih source berdasarkan environment
comment_source = (
    comment_source_local
    if not os.environ.get("MONGODB_CONNECTION_STRING")
    else comment_source_mongo
)


# ── Feature View ──────────────────────────────────────────────────────────────
# Hanya 5 field sesuai skema — tidak boleh ada tambahan lain.
comment_features = FeatureView(
    name="comment_features",
    entities=[comment],
    ttl=timedelta(days=365),
    schema=[
        Field(name="text",          dtype=String),   # teks input mentah
        Field(name="score_toxic",   dtype=Float32),  # skor model 0.0–1.0; NULL untuk bootstrap row
        Field(name="confidence",    dtype=Float32),  # abs(score_toxic - 0.5); NULL untuk bootstrap row
        Field(name="model_version", dtype=String),   # "bootstrap" / "v1" / "v2" / ...
        Field(name="toxic",         dtype=Int32),    # label 0/1; NULL untuk prediction row
    ],
    source=comment_source,
)
