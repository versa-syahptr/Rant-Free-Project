import pandas as pd

BASE = r"C:\Users\Batrisyia Zahrani\Documents\01 ITB\rantfree\jmtcc_datasets"
OUT  = r"C:\Users\Batrisyia Zahrani\Documents\01 ITB\rantfree\services\feature-store\feature_repo\data\dataset.parquet"

# 1. Train (Jigsaw Toxic Comment)
train = pd.read_csv(f"{BASE}\\jigsaw-toxic-comment-train.csv",
                    usecols=["comment_text", "toxic"])

# 2. Unintended Bias — toxic-nya float, binarize >= 0.5
bias = pd.read_csv(f"{BASE}\\jigsaw-unintended-bias-train.csv",
                   usecols=["comment_text", "toxic"])
bias["toxic"] = (bias["toxic"] >= 0.5).astype(int)

# 3. Validation
val = pd.read_csv(f"{BASE}\\validation.csv",
                  usecols=["comment_text", "toxic"])

# 4. Test + Labels (filter baris ignored = -1)
test   = pd.read_csv(f"{BASE}\\test.csv",    usecols=["id", "content"])
labels = pd.read_csv(f"{BASE}\\test_labels.csv", usecols=["id", "toxic"])
test_m = test.merge(labels, on="id")
test_m = test_m[test_m["toxic"] != -1]
test_m = test_m.rename(columns={"content": "comment_text"})[["comment_text", "toxic"]]

# Gabungkan semua
combined = pd.concat([train, bias, val, test_m], ignore_index=True)
combined = combined.rename(columns={"comment_text": "text"})
combined = combined.dropna(subset=["text", "toxic"])
combined["toxic"] = combined["toxic"].astype(int)

print(f"Total baris  : {len(combined):,}")
print(f"Toxic (1)    : {combined['toxic'].sum():,}")
print(f"Non-toxic (0): {(combined['toxic'] == 0).sum():,}")

combined.to_parquet(OUT, index=False)
print(f"\n✅ Tersimpan ke:\n   {OUT}")