from flask import Flask, render_template_string, request, send_file, session
import os
import pandas as pd
import io
from datetime import datetime

import ml_service

MAX_LEN = ml_service.MAX_LEN
MAX_WORDS = ml_service.MAX_WORDS
PREDICTIONS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".predictions_cache")
os.makedirs(PREDICTIONS_CACHE_DIR, exist_ok=True)

_model_info = ml_service.get_model_info()
VAL_ACCURACY = _model_info["val_accuracy"]
MODEL_TRAINED_AT = (
    datetime.fromisoformat(_model_info["trained_at"])
    if _model_info["trained_at"]
    else None
)


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "pi-disaster-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # до 32 МБ для CSV


def _save_predictions_to_cache(csv_content: str) -> str:
    """Сохраняет CSV на диск — не в cookie-сессию (лимит ~4 КБ)."""
    file_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(4).hex()
    path = os.path.join(PREDICTIONS_CACHE_DIR, f"{file_id}.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(csv_content)
    session["predictions_file_id"] = file_id
    return file_id


def _get_predictions_cache_path():
    file_id = session.get("predictions_file_id")
    if not file_id:
        return None
    path = os.path.join(PREDICTIONS_CACHE_DIR, f"{file_id}.csv")
    return path if os.path.isfile(path) else None


def _clear_stored_results():
    """Удаляет все сохранённые результаты: CSV, кэш и единичное предсказание."""
    cache_path = _get_predictions_cache_path()
    if cache_path:
        try:
            os.remove(cache_path)
        except OSError:
            pass
    session.pop("predictions_file_id", None)
    session.pop("predictions_filename", None)
    session.pop("batch_stats", None)
    session.pop("single_result", None)


def _save_single_result(prediction, prob, prob_pct, prob_hue, text):
    session["single_result"] = {
        "prediction": int(prediction),
        "prob": float(prob),
        "prob_pct": float(prob_pct),
        "prob_hue": int(prob_hue),
        "text": text,
    }


def _load_single_result():
    data = session.get("single_result")
    if not data:
        return None, None, None, None, ""
    return (
        data.get("prediction"),
        data.get("prob"),
        data.get("prob_pct"),
        data.get("prob_hue", 120),
        data.get("text", ""),
    )

TEMPLATE = """
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8">
    <title>Классификация твитов о бедствиях</title>
    <style>
      :root {
        --bg-page: #1e1f22;
        --bg-card: #2b2d30;
        --bg-input: #1e1f22;
        --bg-muted: #393b40;
        --border: #4e5157;
        --border-dashed: #5a5d63;
        --text: #dfe1e5;
        --text-secondary: #868a91;
        --text-heading: #bcbec4;
        --accent: #3574f0;
        --accent-hover: #4682fa;
        --danger-bg: #4d2e32;
        --danger-text: #f75464;
        --success-bg: #2d3b2e;
        --success-text: #6aab73;
        --focus-ring: rgba(53, 116, 240, 0.25);
        --btn-bg: #393b40;
        --btn-bg-hover: #43454a;
        --btn-bg-active: #35373b;
        --btn-border: #5a5d63;
        --btn-border-hover: #6f737a;
        --btn-text: #bcbec4;
        --btn-text-hover: #dfe1e5;
      }

      body {
        font-family: "Segoe UI", Arial, sans-serif;
        background: var(--bg-page);
        color: var(--text);
        margin: 0;
        padding: 0;
      }

      .container {
        position: relative;
        max-width: 1100px;
        margin: 40px auto;
        background: var(--bg-card);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px 28px 32px;
      }

      .page-clear-form {
        position: absolute;
        top: 20px;
        right: 20px;
        margin: 0;
        z-index: 5;
      }

      .container.has-clear-btn {
        padding-top: 24px;
      }

      h1 { margin-top: 0; font-size: 24px; color: var(--text); padding-right: 140px; }

      .page-header {
        margin-bottom: 20px;
      }

      .page-header-main .desc {
        margin-bottom: 0;
      }

      p.desc { color: var(--text-secondary); margin-bottom: 20px; }
      p.desc strong { color: var(--success-text); }

      .section-title {
        font-weight: 600;
        margin-top: 22px;
        margin-bottom: 8px;
        color: var(--text-heading);
      }

      .workspace-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px 28px;
        align-items: start;
        margin-top: 8px;
      }

      .workspace-col {
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 0;
      }

      .workspace-col .section-title {
        margin-top: 0;
      }

      .workspace-col textarea {
        min-height: 100px;
      }

      .workspace-col .result-block,
      .workspace-col .batch-stats {
        margin-top: 16px;
      }

      .workspace-col .batch-charts {
        flex-direction: column;
        align-items: stretch;
      }

      .workspace-col .donut-wrap {
        align-self: center;
      }

      @media (max-width: 900px) {
        .workspace-grid {
          grid-template-columns: 1fr;
          gap: 28px;
        }

        .workspace-col + .workspace-col .section-title {
          margin-top: 8px;
        }
      }

      textarea {
        width: 100%;
        min-height: 120px;
        padding: 10px 12px;
        font-size: 14px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--bg-input);
        color: var(--text);
        resize: vertical;
        box-sizing: border-box;
      }

      textarea::placeholder { color: #6f737a; }

      textarea:focus {
        outline: none;
        border-color: var(--accent);
        box-shadow: 0 0 0 2px var(--focus-ring);
      }

      code {
        font-family: Consolas, "Courier New", monospace;
        font-size: 12px;
        color: #6aab73;
        background: rgba(106, 171, 115, 0.12);
        padding: 1px 5px;
        border-radius: 4px;
      }

      .actions {
        margin-top: 16px;
        display: flex;
        justify-content: flex-start;
        gap: 8px;
        align-items: center;
      }

      /* Кнопки — сдержанный тёмный стиль */
      .btn-dg {
        height: 28px;
        padding: 0 16px;
        margin: 0;
        box-sizing: border-box;
        font-family: inherit;
        font-size: 13px;
        font-weight: 500;
        line-height: 26px;
        border-radius: 4px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        background: var(--btn-bg);
        border: 1px solid var(--btn-border);
        color: var(--btn-text);
        transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
        user-select: none;
        white-space: nowrap;
        -webkit-appearance: none;
        appearance: none;
      }

      .btn-dg:hover {
        background: var(--btn-bg-hover);
        border-color: var(--btn-border-hover);
        color: var(--btn-text-hover);
      }

      .btn-dg:active {
        background: var(--btn-bg-active);
        border-color: var(--btn-border);
        color: var(--btn-text-hover);
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.2);
      }

      .btn-dg:focus-visible {
        outline: 1px solid #6f737a;
        outline-offset: 2px;
      }

      .btn-dg.btn-dg-danger {
        background: #8b2942;
        border-color: #c73e5a;
        color: #ffffff;
      }

      .btn-dg.btn-dg-danger:hover {
        background: #a3324f;
        border-color: #f75464;
        color: #ffffff;
      }

      .btn-dg.btn-dg-danger:active {
        background: #6b2033;
        border-color: #a3324f;
        color: #ffffff;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.25);
      }

      .btn-dg-icon {
        width: 14px;
        height: 14px;
        flex-shrink: 0;
        opacity: 0.85;
      }

      .upload {
        margin-top: 8px;
        padding: 12px 14px;
        border-radius: 8px;
        background: var(--bg-muted);
        border: 1px dashed var(--border-dashed);
      }

      .file-picker {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
      }

      /* Полностью скрываем нативный input (в т.ч. «Выберите файл» в Edge/Chrome) */
      .file-picker input[type="file"] {
        display: none !important;
        visibility: hidden !important;
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        pointer-events: none !important;
        clip: rect(0, 0, 0, 0) !important;
        border: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
      }

      .btn-file {
        gap: 8px;
        cursor: pointer;
      }

      .file-name {
        font-size: 12px;
        font-family: Consolas, "Courier New", monospace;
        color: var(--text-secondary);
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .file-name:not(:empty) {
        color: var(--btn-text-hover);
      }

      .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
      }

      .badge-danger { background: var(--danger-bg); color: var(--danger-text); }
      .badge-safe { background: var(--success-bg); color: var(--success-text); }

      .result-block {
        margin-top: 22px;
        padding: 14px 16px;
        border-radius: 10px;
        background: var(--bg-muted);
        border: 1px solid var(--border);
      }

      .label { font-weight: 600; margin-bottom: 4px; color: var(--text); }
      .prob { color: var(--text-secondary); font-size: 14px; }
      .original { margin-top: 10px; font-size: 13px; color: var(--text-secondary); }
      .original span { display: inline-block; margin-top: 4px; color: var(--text); }
      .msg { margin-top: 10px; font-size: 13px; color: var(--text-secondary); }
      .footer { margin-top: 24px; font-size: 11px; color: #6f737a; text-align: right; }

      /* Карточка модели */
      .model-card {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px 16px;
        margin-bottom: 22px;
        padding: 14px 16px;
        border-radius: 10px;
        background: var(--bg-muted);
        border: 1px solid var(--border);
      }

      .model-card-title {
        grid-column: 1 / -1;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 2px;
      }

      .model-stat-label { font-size: 11px; color: var(--text-secondary); }
      .model-stat-value { font-size: 14px; font-weight: 600; color: var(--text); }
      .model-stat-value.accent { color: var(--success-text); }

      /* Шкала вероятности */
      .prob-bar-block { margin-top: 12px; }

      .prob-bar-header {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: var(--text-secondary);
        margin-bottom: 6px;
      }

      .prob-bar-track {
        height: 10px;
        border-radius: 5px;
        background: var(--bg-input);
        border: 1px solid var(--border);
        overflow: hidden;
      }

      .prob-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.35s ease;
        min-width: 2px;
      }

      .prob-bar-caption {
        margin-top: 6px;
        font-size: 12px;
        color: var(--text-secondary);
      }

      /* Статистика по CSV */
      .batch-stats {
        padding: 16px;
        border-radius: 10px;
        background: var(--bg-muted);
        border: 1px solid var(--border);
      }

      .batch-stats h3 {
        margin: 0 0 12px;
        font-size: 14px;
        font-weight: 600;
        color: var(--text-heading);
      }

      .batch-summary {
        font-size: 13px;
        color: var(--text-secondary);
        margin-bottom: 14px;
        line-height: 1.5;
      }

      .batch-summary strong { color: var(--text); }

      .batch-charts {
        display: flex;
        gap: 20px;
        align-items: center;
        flex-wrap: wrap;
      }

      .donut-wrap {
        position: relative;
        width: 88px;
        height: 88px;
        flex-shrink: 0;
      }

      .donut {
        width: 88px;
        height: 88px;
        border-radius: 50%;
        position: relative;
      }

      .donut-hole {
        position: absolute;
        inset: 18px;
        border-radius: 50%;
        background: var(--bg-muted);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
      }

      .batch-bars { flex: 1; min-width: 180px; }

      .bar-row { margin-bottom: 10px; }
      .bar-row:last-child { margin-bottom: 0; }

      .bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        margin-bottom: 4px;
        color: var(--text-secondary);
      }

      .bar-track {
        height: 8px;
        border-radius: 4px;
        background: var(--bg-input);
        border: 1px solid var(--border);
        overflow: hidden;
      }

      .bar-fill-danger { height: 100%; background: var(--danger-text); border-radius: 3px; }
      .bar-fill-safe { height: 100%; background: var(--success-text); border-radius: 3px; }

      .batch-actions {
        margin-top: 14px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
      }

      /* Индикатор загрузки на кнопках */
      .btn-dg.is-loading {
        pointer-events: none;
        opacity: 0.85;
        position: relative;
        padding-left: 32px;
      }

      .btn-dg.is-loading::before {
        content: "";
        position: absolute;
        left: 12px;
        width: 14px;
        height: 14px;
        border: 2px solid var(--btn-border);
        border-top-color: var(--btn-text-hover);
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
      }

      @keyframes spin { to { transform: rotate(360deg); } }

      .page-overlay {
        display: none;
        position: fixed;
        inset: 0;
        background: rgba(30, 31, 34, 0.45);
        z-index: 100;
        align-items: center;
        justify-content: center;
      }

      .page-overlay.active { display: flex; }

      .overlay-box {
        padding: 16px 22px;
        border-radius: 8px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        font-size: 13px;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .overlay-spinner {
        width: 20px;
        height: 20px;
        border: 2px solid var(--border);
        border-top-color: var(--accent);
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
        flex-shrink: 0;
      }
    </style>
  </head>
  <body>
    <div class="page-overlay" id="page-overlay" aria-hidden="true">
      <div class="overlay-box">
        <div class="overlay-spinner"></div>
        <span id="overlay-text">Выполняется предсказание…</span>
      </div>
    </div>

    <div class="container{% if batch_stats or prediction is not none %} has-clear-btn{% endif %}">
      {% if batch_stats or prediction is not none %}
      <form method="post" class="page-clear-form">
        <input type="hidden" name="action" value="clear">
        <button type="submit" class="btn-dg btn-dg-danger">
          Очистить данные
        </button>
      </form>
      {% endif %}

      <div class="page-header">
        <div class="page-header-main">
          <h1>Классификация твитов: бедствие или нет</h1>
          <p class="desc">
            Модель, обученная на датасете Twitter Disaster, определяет, описывает ли текст реальное бедствие.
          </p>
        </div>
      </div>

      {% if file_message and prediction is none and not batch_stats %}
      <div class="msg">{{ file_message }}</div>
      {% endif %}

      <div class="model-card">
        <div class="model-card-title">Модель</div>
        <div>
          <div class="model-stat-label">Архитектура</div>
          <div class="model-stat-value">Embedding + LSTM</div>
        </div>
        <div>
          <div class="model-stat-label">Словарь / длина</div>
          <div class="model-stat-value">{{ max_words }} слов · {{ max_len }} токенов</div>
        </div>
        <div>
          <div class="model-stat-label">Accuracy (валидация)</div>
          <div class="model-stat-value accent">
            {% if val_accuracy is not none %}{{ (val_accuracy * 100)|round(2) }}%{% else %}—{% endif %}
          </div>
        </div>
        <div>
          <div class="model-stat-label">Обучена</div>
          <div class="model-stat-value">
            {% if model_trained_at %}{{ model_trained_at.strftime("%d.%m.%Y %H:%M") }}{% else %}—{% endif %}
          </div>
        </div>
      </div>

      <div class="workspace-grid">
        <div class="workspace-col">
          <div class="section-title">1. Единичный твит</div>
          <form method="post" class="js-loading-form" data-loading-text="Анализ твита…">
            <input type="hidden" name="action" value="single">
            <textarea name="text" placeholder="Например: Massive earthquake just hit the city, buildings collapsed.">{{ text or "" }}</textarea>
            <div class="actions">
              <button type="submit" class="btn-dg js-submit-btn">
                Сделать предсказание
              </button>
            </div>
          </form>

          {% if single_message %}
          <div class="msg">{{ single_message }}</div>
          {% endif %}

          {% if prediction is not none %}
          <div class="result-block">
            <div class="label">
              Результат:
              {% if prediction == 1 %}
                <span class="badge badge-danger">БЕДСТВИЕ</span>
              {% else %}
                <span class="badge badge-safe">НЕТ БЕДСТВИЯ</span>
              {% endif %}
            </div>
            <div class="prob-bar-block">
              <div class="prob-bar-header">
                <span>Нет бедствия</span>
                <span>{{ prob_pct }}%</span>
                <span>Бедствие</span>
              </div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: {{ prob_pct }}%; background: hsl({{ prob_hue }}, 45%, 42%);"></div>
              </div>
              <div class="prob-bar-caption">
                Вероятность бедствия: <strong>{{ prob|round(4) }}</strong> (порог: 0.5)
              </div>
            </div>
            <div class="original">
              Исходный текст:
              <span>{{ text }}</span>
            </div>
          </div>
          {% endif %}
        </div>

        <div class="workspace-col">
          <div class="section-title">2. Загрузка файла test.csv</div>
          <form method="post" enctype="multipart/form-data" class="js-loading-form" data-loading-text="Обработка файла…">
            <input type="hidden" name="action" value="file">
            <div class="upload">
              <div class="file-picker">
                <input type="file" id="file-input" name="file" accept=".csv" tabindex="-1" aria-hidden="true">
                <button type="button" class="btn-dg btn-file" id="file-btn">
                  <svg class="btn-dg-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
                    <path d="M9 1.5H4.5A1.5 1.5 0 0 0 3 3v10a1.5 1.5 0 0 0 1.5 1.5h7A1.5 1.5 0 0 0 13 13V6L9 1.5Z"/>
                    <path d="M9 1.5V6H13"/>
                    <path d="M6 9.5h4M6 11.5h2.5"/>
                  </svg>
                  Выбрать файл
                </button>
                <span class="file-name" id="file-name"></span>
              </div>
              <div class="msg">Ожидается CSV с колонкой <code>text</code> (и, при желании, <code>id</code>).</div>
            </div>
            <div class="actions">
              <button type="submit" class="btn-dg js-submit-btn">
                Запустить предсказания для файла
              </button>
            </div>
          </form>

          {% if file_message and not batch_stats and prediction is none %}
          <div class="msg">{{ file_message }}</div>
          {% endif %}

          {% if file_message and batch_stats %}
          <div class="msg">{{ file_message }}</div>
          {% endif %}

          {% if batch_stats %}
          <div class="batch-stats">
            <h3>Результаты: {{ batch_stats.filename }}</h3>
            <p class="batch-summary">
              Обработано <strong>{{ batch_stats.total }}</strong> записей:
              <strong style="color: var(--danger-text);">{{ batch_stats.disaster }}</strong> бедствий
              ({{ batch_stats.disaster_pct }}%),
              <strong style="color: var(--success-text);">{{ batch_stats.safe }}</strong> обычных
              ({{ batch_stats.safe_pct }}%).
            </p>
            <div class="batch-charts">
              <div class="donut-wrap">
                <div class="donut" style="background: conic-gradient(
                  var(--danger-text) 0% {{ batch_stats.disaster_pct }}%,
                  var(--success-text) {{ batch_stats.disaster_pct }}% 100%
                );">
                  <div class="donut-hole">{{ batch_stats.total }}</div>
                </div>
              </div>
              <div class="batch-bars">
                <div class="bar-row">
                  <div class="bar-label">
                    <span>Бедствие</span>
                    <span>{{ batch_stats.disaster }} ({{ batch_stats.disaster_pct }}%)</span>
                  </div>
                  <div class="bar-track">
                    <div class="bar-fill-danger" style="width: {{ batch_stats.disaster_pct }}%;"></div>
                  </div>
                </div>
                <div class="bar-row">
                  <div class="bar-label">
                    <span>Нет бедствия</span>
                    <span>{{ batch_stats.safe }} ({{ batch_stats.safe_pct }}%)</span>
                  </div>
                  <div class="bar-track">
                    <div class="bar-fill-safe" style="width: {{ batch_stats.safe_pct }}%;"></div>
                  </div>
                </div>
              </div>
            </div>
            <div class="batch-actions">
              <form method="post" style="display:inline;">
                <input type="hidden" name="action" value="download">
                <button type="submit" class="btn-dg">
                  Скачать CSV с предсказаниями
                </button>
              </form>
            </div>
          </div>
          {% endif %}
        </div>
      </div>

      <div class="footer">
        Flask‑интерфейс к модели Keras (Twitter Disaster).
      </div>
    </div>

    <script>
      (function () {
        var input = document.getElementById("file-input");
        var btn = document.getElementById("file-btn");
        var nameEl = document.getElementById("file-name");
        if (input && btn) {
          btn.addEventListener("click", function () { input.click(); });
          input.addEventListener("change", function () {
            nameEl.textContent = input.files[0] ? input.files[0].name : "";
          });
        }

        var overlay = document.getElementById("page-overlay");
        var overlayText = document.getElementById("overlay-text");

        function hideOverlay() {
          if (overlay) {
            overlay.classList.remove("active");
            overlay.setAttribute("aria-hidden", "true");
          }
          document.querySelectorAll(".btn-dg.is-loading").forEach(function (b) {
            b.classList.remove("is-loading");
            b.disabled = false;
          });
        }

        hideOverlay();
        window.addEventListener("pageshow", hideOverlay);

        document.querySelectorAll(".js-loading-form").forEach(function (form) {
          form.addEventListener("submit", function () {
            var submitBtn = form.querySelector(".js-submit-btn");
            var text = form.getAttribute("data-loading-text") || "Обработка…";
            if (overlay) {
              overlay.classList.add("active");
              overlay.setAttribute("aria-hidden", "false");
              if (overlayText) overlayText.textContent = text;
            }
            if (submitBtn) {
              submitBtn.classList.add("is-loading");
              setTimeout(function () { submitBtn.disabled = true; }, 0);
            }
          });
        });
      })();
    </script>
  </body>
  </html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction, prob, prob_pct, prob_hue, text = _load_single_result()
    if prediction is None:
        prob = None
        prob_pct = None
        prob_hue = 120
        text = ""
    file_message = ""
    single_message = ""
    batch_stats = session.get("batch_stats")

    if request.method == "POST":
        action = request.form.get("action")

        if action == "download":
            cache_path = _get_predictions_cache_path()
            filename = session.get("predictions_filename", "predictions.csv")
            if cache_path:
                return send_file(
                    cache_path,
                    as_attachment=True,
                    download_name=filename,
                    mimetype="text/csv",
                )
            file_message = "Нет сохранённых предсказаний. Загрузите CSV заново."

        elif action == "clear":
            _clear_stored_results()
            batch_stats = None
            prediction = None
            prob = None
            prob_pct = None
            prob_hue = 120
            text = ""
            file_message = "Данные очищены."

        # 1) Единичный твит
        elif action == "single":
            text = request.form.get("text", "").strip()
            if text:
                result = ml_service.predict_text(text)
                prediction = result["label"]
                prob = result["probability"]
                prob_pct = round(min(prob * 100, 100), 1)
                prob_hue = int((1 - prob) * 120)
                _save_single_result(prediction, prob, prob_pct, prob_hue, text)
            else:
                single_message = "Введите текст твита."
        elif action == "file":
            uploaded = request.files.get("file")
            if uploaded and uploaded.filename.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded)
                    if "text" not in df.columns:
                        file_message = "В CSV не найдена колонка 'text'."
                    else:
                        result_df, summary = ml_service.predict_dataframe(df)

                        output = io.StringIO()
                        result_df.to_csv(output, index=False)
                        csv_content = output.getvalue()

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"predictions_{timestamp}.csv"

                        batch_stats = {
                            **summary,
                            "filename": uploaded.filename,
                        }
                        session["predictions_filename"] = filename
                        session["batch_stats"] = batch_stats
                        _save_predictions_to_cache(csv_content)
                except Exception as e:
                    file_message = f"Ошибка при обработке файла: {e}"
            else:
                file_message = "Пожалуйста, выберите CSV-файл."

    return render_template_string(
        TEMPLATE,
        prediction=prediction,
        prob=prob,
        prob_pct=prob_pct,
        prob_hue=prob_hue,
        text=text,
        file_message=file_message,
        single_message=single_message,
        batch_stats=batch_stats,
        val_accuracy=VAL_ACCURACY,
        max_words=MAX_WORDS,
        max_len=MAX_LEN,
        model_trained_at=MODEL_TRAINED_AT,
    )


if __name__ == "__main__":
    # Запуск локального сервера
    app.run(host="0.0.0.0", port=5000, debug=True)
