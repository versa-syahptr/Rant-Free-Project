"""
bootstrap.py — Rant-Free Feature Store
Memuat data/dataset.parquet (ground truth Jigsaw Multilingual) ke Feast
sebagai "bootstrap rows" sebelum ada traffic live.

Jalankan sekali sebelum Model Service mulai berjalan:
    python bootstrap.py

Untuk testing lokal (tanpa mongoDB):
    python bootstrap.py
    (otomatis pakai feature_store.local.yaml)
"""

import os
import uuid
import pandas as pd
from datetime import datetime, timezone
import pyarrow as pa
from feast import FeatureStore
from pymongo import MongoClient
from store import get_store
from feature_definitions import comment, comment_features
from feast.infra.offline_stores.contrib.mongodb_offline_store.mongodb import MongoDBOfflineStore

# ── Konfigurasi ───────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_PATH    = os.getenv("FEATURE_STORE_REPO_PATH", BASE_DIR)
YAML_FILE    = os.getenv("FEATURE_STORE_YAML", os.path.join(BASE_DIR, "feature_store.local.yaml"))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.parquet")
OUTPUT_PATH  = os.path.join(BASE_DIR, "data", "dataset-output.parquet")



def build_bootstrap_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konversi dataset Jigsaw Multilingual ke format bootstrap rows.

    Bootstrap row spec:
      request_id      → UUID baru
      text            → teks mentah
      score_toxic     → NULL (belum ada model)
      confidence      → NULL (belum ada model)
      model_version   → "bootstrap"
      toxic           → ground truth 0/1 dari dataset
      event_timestamp → now()
    """
    now = pd.Timestamp(datetime.now(timezone.utc))

    result = pd.DataFrame({
        "request_id":      [str(uuid.uuid4()) for _ in range(len(df))],
        "event_timestamp": now,
        "text":            df["text"].values,
        "language":        df["lang"].values,
        "score_toxic":  float("nan"),
        "confidence":   float("nan"),
        "model_version":   "bootstrap",
        "toxic":           df["toxic"].astype("Int32").values,  # nullable int
    })
    result = result.astype({
        "score_toxic": "float32",
        "confidence":  "float32",
        "toxic":       "Int32",
    })

    return result


def bootstrap_direct_write(store: FeatureStore, df: pd.DataFrame):
    """
    Bypass write_to_offline_store karena Feast MongoDB butuh sample dokumen
    untuk infer schema — chicken-and-egg problem saat collection masih kosong.
    Setelah bootstrap selesai, write_to_offline_store bisa dipakai normal
    oleh Model Service dan HITL Backend.
    """
    total = len(df)
    written = 0

    def log_progress(n: int):
        nonlocal written
        written += n
        print(f"  {written:,} / {total:,} baris ({written/total*100:.1f}%)")

    MongoDBOfflineStore.offline_write_batch(
        config=store.config,
        feature_view=store.get_feature_view("comment_features"),
        table=pa.Table.from_pandas(df),
        progress=log_progress,
    )

def is_bootstrapped(client: MongoClient, database: str, collection: str) -> bool:
    meta = client[database][collection].find_one({"feature_view": "__bootstrap_meta"})
    return meta is not None


def mark_bootstrapped(client: MongoClient, database: str, collection: str, row_count: int) -> None:
    client[database][collection].insert_one({
        "feature_view": "__bootstrap_meta",
        "completed_at": datetime.now(timezone.utc),
        "row_count": row_count,
    })

def start_bootstrap(forced: bool = False):
    print("=== Rant-Free Bootstrap ===\n")

    if os.environ.get("MONGODB_CONNECTION_STRING"):
        mongo_conn = os.environ["MONGODB_CONNECTION_STRING"]
        mongo_db   = os.environ.get("MONGODB_DATABASE", "feast")
        mongo_col  = os.environ.get("MONGODB_COLLECTION", "feature_history")

        client = MongoClient(mongo_conn)
        if not forced and is_bootstrapped(client, mongo_db, mongo_col):
            print("Bootstrap sudah pernah dijalankan sebelumnya — lewati")
            return
    else:
        print("Mode lokal — tidak ada mongoDB, skip cek bootstrap sebelumnya\n")

    # 1. Load dataset
    print(f"Membaca dataset dari: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    assert "text"  in df.columns, "Kolom 'text' tidak ditemukan — jalankan merge_dataset.py dulu"
    assert "toxic" in df.columns, "Kolom 'toxic' tidak ditemukan"
    print(f"  Total baris  : {len(df):,}")
    print(f"  Toxic (1)    : {df['toxic'].sum():,}")
    print(f"  Non-toxic (0): {(df['toxic'] == 0).sum():,}")

    # 2. Bangun bootstrap rows
    print("\nMembangun bootstrap rows...")
    bootstrap_df = build_bootstrap_rows(df)
    print(f"  Kolom: {list(bootstrap_df.columns)}")
    print(f"  Sample request_id: {bootstrap_df['request_id'].iloc[0]}")

    # 3. Simpan ke parquet (overwrite — kolom sekarang lengkap dengan request_id)
    bootstrap_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nTersimpan ke: {OUTPUT_PATH}")

    # 4. Feast apply — daftarkan entity & feature view ke registry
    # print(f"\nMenjalankan FeatureStore dari: {REPO_PATH} (yaml: {YAML_FILE})")
    # store = FeatureStore(repo_path=REPO_PATH, fs_yaml_file=YAML_FILE)
    store = get_store()

    store.apply([comment, comment_features])
    print("feast apply OK — entity & feature view terdaftar\n")

    # 5. write to offline store — load bootstrap rows ke mongoDB (jika pakai postgres)
    if os.environ.get("MONGODB_CONNECTION_STRING"):
        print("Menulis bootstrap rows ke mongoDB...")
        bootstrap_direct_write(store, bootstrap_df)
        print(f"  {len(bootstrap_df):,} baris ditulis\n")
        mark_bootstrapped(client, mongo_db, mongo_col, len(bootstrap_df))
    else:
        print("Mode lokal — data dibaca langsung dari parquet, skip write\n")


    print("\nBootstrap selesai!")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Bootstrap Rant-Free Feature Store dengan dataset Jigsaw Multilingual")
    parser.add_argument("--force", action="store_true", help="Jalankan bootstrap meski sudah pernah dijalankan sebelumnya")
    args = parser.parse_args()

    start_bootstrap(forced=args.force)