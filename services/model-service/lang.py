# Originally made by Rani

import os
import urllib.request
import fasttext

MODEL_PATH = "data/lid.176.bin"

# Download model sekali saja
if not os.path.exists(MODEL_PATH):
    print("Downloading fasttext language model (~126MB)...")
    urllib.request.urlretrieve(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        MODEL_PATH
    )

model = fasttext.load_model(MODEL_PATH)

# Deteksi bahasa dengan fasttext-wheel
def detect_lang_with_fasttext(text):
    try:
        prediction = model.predict(str(text).replace("\n", " "))
        prediction = prediction[0]
        if len(prediction) == 0:
            return "unknown"
        
        label = prediction[0]
        return label.replace("_label_", "")
    
    except Exception as e:
        print(e)
        return "unknown"
