import os
from pathlib import Path
import sys
import time

USE_LOCAL = os.getenv("USE_LOCAL", "false") == "true"
if USE_LOCAL:
    print("USE_LOCAL activated!")
    # Possibly ... should import manually
    script_path = Path(__file__).resolve()
    target_path = (script_path.parent / "..").resolve()
    sys.path.append(str(target_path))

from feature_store.reader import read_training_data   # type: ignore

import pandas as pd

def sample() -> pd.DataFrame:
    # I think I need to know how to use Feast.
    # Basically, I want to take some N representative data, let's say 4000.
    # 2000 of them are toxic, 2000 is not
    print(f"Calling feature_store.read_training_data ....")
    start_time = time.perf_counter()
    df = read_training_data()
    duration = time.perf_counter() - start_time
    print(f"(feature_store.read_training_data duration: {duration} s)")

    toxic_df = df[df["toxic"] == 1].sample(2000, random_state=120)
    nontoxic_df = df[df["toxic"] == 0].sample(2000, random_state=120)
    return pd.concat([toxic_df, nontoxic_df])

if __name__ == "__main__":
    print("Reading ....")
    start_time = time.perf_counter()
    df = sample()
    duration = time.perf_counter() - start_time
    print("DataFrame size:", len(df))
    print(f"(feature_store.read_training_data duration: {duration} s)")
