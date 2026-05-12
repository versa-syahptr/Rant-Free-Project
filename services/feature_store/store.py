import os
from feast import FeatureStore
from feast.repo_config import RepoConfig
from feast.infra.offline_stores.contrib.mongodb_offline_store.mongodb import MongoDBOfflineStoreConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_store() -> FeatureStore:
    mongo_conn = os.environ.get("MONGODB_CONNECTION_STRING")

    registry_path_dir = os.environ.get(
        "FEATURE_STORE_REGISTRY_PATH",
        os.path.join(BASE_DIR, "data"),  # fallback untuk local dev
    )
    registry_file = os.path.join(registry_path_dir, "registry.db")

    if mongo_conn:
        config = RepoConfig(
            project="rant_free",
            registry=registry_file,
            provider="local",
            offline_store=MongoDBOfflineStoreConfig(
                connection_string=mongo_conn,
                database=os.environ.get("MONGODB_DATABASE", "feast"),
                collection=os.environ.get("MONGODB_COLLECTION", "feature_history"),
            ),
            online_store={"type": "sqlite", "path": "/tmp/online_store.db"},
            entity_key_serialization_version=2,
        )
        return FeatureStore(config=config)
    else:
        return FeatureStore(repo_path=BASE_DIR)