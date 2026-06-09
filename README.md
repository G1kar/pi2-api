# Классификация твитов о бедствиях (Twitter Disaster NLP)

Веб-приложение и модель машинного обучения для бинарной классификации коротких текстов: определение постов из Twitter, описывающих **реальное бедствие**, от обычных сообщений.

**Сроки проекта:** 01.09.2025 — 01.02.2026  

**Система планирования:** канбан-доска (этапы и задачи описаны в [PROJECT.md](PROJECT.md)).

---

## Состав команды и роли

| Участник | Роли |
|----------|------|
| **Казимиров Никита** | Программист, Scrum-мастер |
| **Хромушин Станислав** | Технический писатель |
| **Попов Константин** | Программист, Архитектор, Тестировщик |

---

## Технологии

- **Python 3**
- **TensorFlow / Keras** — нейросетевая модель (Embedding + LSTM)
- **Flask** — веб-интерфейс
- **FastAPI** — REST API для предсказаний
- **pandas, scikit-learn** — данные и разбиение выборок

---

## Структура проекта

```
PI/
├── README.md           # этот файл
├── PROJECT.md          # план проекта, канбан, задачи по ролям
├── train.py            # обучение модели на train.csv
├── test.py             # предсказания для test.csv (офлайн)
├── app.py              # веб-интерфейс (форма + загрузка CSV)
├── api.py              # REST API (FastAPI)
├── ml_service.py       # общая логика модели для Flask и FastAPI
├── requirements.txt    # зависимости Python
├── train.csv           # обучающая выборка (id, keyword, location, text, target)
├── test.csv            # тестовая выборка (id, keyword, location, text)
├── disaster_model.h5   # сохранённая модель Keras
├── tokenizer.pkl       # токенайзер (сохраняется после train.py)
└── metrics.txt         # метрика accuracy на валидации
```

---

## Установка и запуск

### 1. Зависимости

```bash
pip install -r requirements.txt
```

### 2. Обучение модели

```bash
python train.py
```

Скрипт читает `train.csv`, обучает модель, сохраняет `disaster_model.h5`, `tokenizer.pkl` и записывает accuracy в `metrics.txt`.

### 3. Предсказания по файлу test.csv (офлайн)

```bash
python test.py
```

Создаёт `test_predictions.csv` и при наличии колонки `id` — `submission.csv`.

### 4. Веб-интерфейс

```bash
python app.py
```

В браузере открыть: **http://127.0.0.1:5000**

- **Единичный твит** — ввод текста, получение предсказания (бедствие / нет) и вероятности.
- **Загрузка CSV** — загрузка файла с колонкой `text`; скачивание CSV с колонками `prediction` и `probability`.
- На странице отображается **Accuracy на валидации** (из `metrics.txt`).

### 5. REST API (FastAPI)

```bash
uvicorn api:app --reload --port 8000
```

Документация Swagger: **http://127.0.0.1:8000/docs**

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/health` | Проверка доступности сервиса |
| `GET` | `/model/info` | Архитектура модели, accuracy, дата обучения |
| `POST` | `/predict` | Один твит → метка и вероятность |
| `POST` | `/predict/batch` | Список текстов → список предсказаний |
| `POST` | `/predict/csv` | CSV с колонкой `text` → JSON или файл (`?download=true`) |

Примеры:

```bash
curl http://127.0.0.1:8000/health

curl http://127.0.0.1:8000/model/info

curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"Massive earthquake hit the city\"}"

curl -X POST http://127.0.0.1:8000/predict/batch ^
  -H "Content-Type: application/json" ^
  -d "{\"texts\": [\"fire in building\", \"nice weather today\"]}"

curl -X POST "http://127.0.0.1:8000/predict/csv?download=true" ^
  -F "file=@test.csv" -o predictions.csv
```

Flask (порт 5000) и FastAPI (порт 8000) можно запускать одновременно — оба используют `ml_service.py`.

---

## Формат данных

- **train.csv**: колонки `id`, `keyword`, `location`, `text`, `target` (0 — не бедствие, 1 — бедствие).
- **test.csv**: те же колонки, кроме `target`; для веб-загрузки достаточно колонки `text` (и при необходимости `id`).

---

## Метрики

- **Accuracy** — доля правильных ответов на валидационной выборке; вычисляется в `train.py`, сохраняется в `metrics.txt` и отображается в веб-интерфейсе.

---

## Лицензия и использование

Учебный проект. Датасет — открытые данные (Kaggle NLP Disaster Tweets и аналоги).
