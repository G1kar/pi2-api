import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers


# ---------- Загрузка train.csv ----------
train_df = pd.read_csv("train.csv")


def clean_text(s: str) -> str:
    """Простая очистка текста твита."""
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"@[A-Za-z0-9_]+", " ", s)
    s = re.sub(r"#", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


train_df = train_df.dropna(subset=["text", "target"])
texts = train_df["text"].astype(str).apply(clean_text).values
labels = train_df["target"].astype(int).values

# ---------- train / val ----------
X_train, X_val, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.2,  # 20% на валидацию
    random_state=42,
    stratify=labels,
)

MAX_WORDS = 20000
MAX_LEN = 40

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(
    X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post"
)
X_val_pad = pad_sequences(
    X_val_seq, maxlen=MAX_LEN, padding="post", truncating="post"
)

y_train = np.array(y_train)
y_val = np.array(y_val)

# ---------- модель ----------
EMBED_DIM = 64

model = models.Sequential(
    [
        layers.Embedding(
            input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN
        ),
        layers.SpatialDropout1D(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

model.fit(
    X_train_pad,
    y_train,
    batch_size=64,
    epochs=10,
    validation_data=(X_val_pad, y_val),
    callbacks=[early_stopping],
)

# ---------- оценка точности (accuracy) на валидации ----------
_, val_acc = model.evaluate(X_val_pad, y_val, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")

with open("metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"val_accuracy={val_acc:.6f}\n")

# ---------- сохранение модели и токенайзера ----------

model.save("disaster_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Модель сохранена в 'disaster_model.h5'")
print("Токенайзер сохранён в 'tokenizer.pkl'")