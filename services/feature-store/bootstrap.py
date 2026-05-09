"""
bootstrap.py — Rant-Free Feature Store
Memuat data/dataset.parquet (ground truth Jigsaw Multilingual) ke Feast
sebagai "bootstrap rows" sebelum ada traffic live.

Jalankan sekali sebelum Model Service mulai berjalan:
    python bootstrap.py

Untuk testing lokal (tanpa PostgreSQL):
    python bootstrap.py
    (otomatis pakai feature_store.local.yaml)
"""

import os
import uuid
import pandas as pd
from datetime import datetime, timezone
from feast import FeatureStore
from feature_definitions import comment, comment_features

# ── Konfigurasi ───────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_PATH    = os.getenv("FEATURE_STORE_REPO_PATH", BASE_DIR)
YAML_FILE    = os.getenv("FEATURE_STORE_YAML", os.path.join(BASE_DIR, "feature_store.local.yaml"))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.parquet")
OUTPUT_PATH  = os.path.join(BASE_DIR, "data", "dataset.parquet")


def compute_confidence(score_toxic: float) -> float:
    """
    Dihitung oleh Model Service, disimpan apa adanya.
    0.0 = paling tidak yakin (skor mendekati 0.5)
    0.5 = paling yakin (skor mendekati 0.0 atau 1.0)
    Dikali 2 agar range [0, 1] dan threshold tetap 0.5.
    """
    return abs(score_toxic - 0.5) * 2


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
        "score_toxic":     None,         # NULL — belum ada model
        "confidence":      None,         # NULL — belum ada model
        "model_version":   "bootstrap",
        "toxic":           df["toxic"].astype("Int32").values,  # nullable int
    })

    return result


def main():
    print("=== Rant-Free Bootstrap ===\n")

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
    print(f"\nMenjalankan FeatureStore dari: {REPO_PATH} (yaml: {YAML_FILE})")
    store = FeatureStore(repo_path=REPO_PATH, fs_yaml_file=YAML_FILE)

    store.apply([comment, comment_features])
    print("feast apply OK — entity & feature view terdaftar\n")

    # 5. Verifikasi baca dari offline store (sanity check 5 baris)
    sample_ids = bootstrap_df["request_id"].head(5).tolist()
    sample_ts  = [bootstrap_df["event_timestamp"].iloc[0]] * 5

    entity_df = pd.DataFrame({
        "request_id":      sample_ids,
        "event_timestamp": sample_ts,
    })

    result = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "comment_features:text",
            "comment_features:score_toxic",
            "comment_features:confidence",
            "comment_features:model_version",
            "comment_features:toxic",
        ],
    ).to_df()

    print("=== Sanity Check — 5 Baris Pertama ===")
    print(result[["request_id", "model_version", "toxic", "score_toxic"]].to_string(index=False))

    null_count = result["score_toxic"].isna().sum()
    print(f"\n✅ score_toxic NULL: {null_count}/5 (harusnya 5/5 untuk bootstrap)")
    print(f"✅ model_version   : {result['model_version'].unique().tolist()} (harusnya ['bootstrap'])")
    print("\nBootstrap selesai!")


if __name__ == "__main__":
    main()