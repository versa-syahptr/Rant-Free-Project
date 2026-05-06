from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import String, Float32, Int64

# Entity → "kunci" utama data kita
comment = Entity(
    name="comment",
    join_keys=["id"],
)

# Source → file parquet yang tadi kita buat
comment_source = FileSource(
    path=r"C:\Users\Batrisyia Zahrani\Documents\01 ITB\rantfree\feature_repo\feature_repo\data\comment_features.parquet",
    event_timestamp_column="event_timestamp",
)

# Feature View → kumpulan fitur untuk entity comment
comment_features_view = FeatureView(
    name="comment_features",
    entities=[comment],
    ttl=timedelta(days=365),
    schema=[
        Field(name="text_length",        dtype=Int64),
        Field(name="word_count",         dtype=Int64),
        Field(name="exclamation_count",  dtype=Int64),
        Field(name="question_count",     dtype=Int64),
        Field(name="uppercase_ratio",    dtype=Float32),
        Field(name="toxic_word_count",   dtype=Int64),
    ],
    source=comment_source,
)