# Note for all services that uses feature store!
- model-service
- training-service
- hitl-backend
- monitoring-service

## Install `services/feature_store/` as an editable package first!
```
pip install -e services/feature_store
```

## Import it to use the store

```python
from feature_store.store import get_store

store = get_store() # instance of feast.FeatureStore
```

## How to actually write to offline store

```python
toxic = score_toxic >= 0.5

df = pd.DataFrame({
    "request_id":      [request_id],
    "event_timestamp": [pd.Timestamp(datetime.now(timezone.utc))],
    "text":            [text],
    "language":        [language],
    "score_toxic":     [score_toxic],
    "confidence":      [confidence],
    "model_version":   [model_version],
    "toxic":           [toxic],
})

# actually write to feast
store.write_to_offline_store(
    feature_view_name="comment_features",
    df=df,
)

```

## how to fetch all data from offline store

```python
from feature_store.reader import read_training_data

df = read_training_data()  # this will fetch all data
# do sampling on your own
```
