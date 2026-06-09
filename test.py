import re
import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Должно совпадать с train.py
MAX_LEN = 40


def clean_text(s: str) -> str:
    """Та же функция очистки, что и в train.py."""
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@[A-Za-z0-9_]+", " ", s)
    s = re.sub(r"#", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- Загрузка обученной модели и токенайзера ----------

model = load_model("disaster_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# ---------- Загрузка test.csv и предсказания ----------

test_df = pd.read_csv("test.csv")  # в test.csv обычно нет target

test_texts = test_df["text"].astype(str).apply(clean_text).values
test_seq = tokenizer.texts_to_sequences(test_texts)
test_pad = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

test_proba = model.predict(test_pad)
test_pred = (test_proba >= 0.5).astype(int)

# Сохраняем предсказания
test_df["prediction"] = test_pred
test_df.to_csv("test_predictions.csv", index=False)
print("Файл с предсказаниями сохранён как 'test_predictions.csv'")

# Дополнительно (пригодится для Kaggle): id + target
if "id" in test_df.columns:
    submission = pd.DataFrame(
        {"id": test_df["id"], "target": test_pred.reshape(-1)}
    )
    submission.to_csv("submission.csv", index=False)
    print("Файл 'submission.csv' (id, target) сохранён для отправки.")