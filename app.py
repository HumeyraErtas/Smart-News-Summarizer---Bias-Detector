import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline
from newspaper import Article

app = Flask(__name__)

DB_PATH = "news.db"

# ---------------------------
# 1) VERİTABANI YARDIMCI FONKSİYONLARI
# ---------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            source_type TEXT,        -- 'url' veya 'text'
            source_input TEXT,       -- girilen url veya text'in ilk 200 karakteri
            summary TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            bias_score INTEGER,
            bias_label TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def save_analysis(source_type, source_input, summary,
                  sentiment_label, sentiment_score,
                  bias_score, bias_label):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO analyses (
            created_at, source_type, source_input,
            summary, sentiment_label, sentiment_score,
            bias_score, bias_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            source_type,
            source_input[:200],   # Çok uzunsa kısalt
            summary,
            sentiment_label,
            sentiment_score,
            bias_score,
            bias_label,
        ),
    )
    conn.commit()
    conn.close()


def get_last_analyses(limit=5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT id, created_at, source_type, source_input,
               summary, sentiment_label, sentiment_score,
               bias_score, bias_label
        FROM analyses
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


# ---------------------------
# 2) HUGGINGFACE MODELLERİ
# ---------------------------

# Özetleme modeli (İlk sürüm: İngilizce haberler için)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# Duygu analizi modeli
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)


# ---------------------------
# 3) HABER METNİNİ URL'DEN ÇEKME
# ---------------------------

def get_article_text_from_url(url: str) -> str:
    article = Article(url)
    article.download()
    article.parse()
    return article.text


# ---------------------------
# 4) BIAS SKORU HESAPLAMA (BASİT HEURİSTİK)
# ---------------------------

def compute_bias(sentiment_label: str, sentiment_score: float):
    """
    Çok basit bir mantık:

    - Pozitif veya negatif duygu güçlü ise -> yüksek bias (ör: 70-100)
    - Nötr ise -> düşük bias (ör: 0-30)

    Gerçek hayatta bu konu çok daha zor, bu sadece proje için basit bir demo.
    """
    label_lower = sentiment_label.lower()

    if "positive" in label_lower:
        base = 70
        bias_label = "Muhtemel pozitif taraflı (pro)"
    elif "negative" in label_lower:
        base = 70
        bias_label = "Muhtemel negatif taraflı (anti)"
    else:
        base = 20
        bias_label = "Göreceli olarak tarafsız"

    # Model confidence (0–1 arası) skoruyla biraz oyna
    bias_score = int(min(100, max(0, base + (sentiment_score - 0.5) * 60)))
    return bias_score, bias_label


# ---------------------------
# 5) FLASK ROUTES
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error_message = None

    if request.method == "POST":
        input_type = request.form.get("input_type")  # 'url' veya 'text'
        url = request.form.get("url") or ""
        raw_text = request.form.get("raw_text") or ""

        try:
            # 1) Haber metnini al
            if input_type == "url" and url.strip():
                full_text = get_article_text_from_url(url.strip())
                source_type = "url"
                source_input = url.strip()
            elif input_type == "text" and raw_text.strip():
                full_text = raw_text.strip()
                source_type = "text"
                source_input = raw_text.strip()
            else:
                raise ValueError("Lütfen URL veya haber metni girin.")

            if len(full_text) < 50:
                raise ValueError("Haber metni çok kısa. Lütfen daha uzun bir içerik deneyin.")

            # 2) Özet üret
            # BART için max_length/min_length ayarları:
            summary_chunks = summarizer(
                full_text,
                max_length=150,
                min_length=40,
                do_sample=False
            )
            summary_text = summary_chunks[0]["summary_text"]

            # 3) Duygu / ton analizi
            sentiment_result = sentiment_model(summary_text)[0]
            sentiment_label = sentiment_result["label"]
            sentiment_score = float(sentiment_result["score"])

            # 4) Bias skorunu hesapla
            bias_score, bias_label = compute_bias(sentiment_label, sentiment_score)

            # 5) Veritabanına kaydet
            save_analysis(
                source_type=source_type,
                source_input=source_input,
                summary=summary_text,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                bias_score=bias_score,
                bias_label=bias_label,
            )

            result = {
                "full_text": full_text,
                "summary": summary_text,
                "sentiment_label": sentiment_label,
                "sentiment_score": round(sentiment_score, 3),
                "bias_score": bias_score,
                "bias_label": bias_label,
                "source_type": source_type,
                "source_input": source_input,
            }

        except Exception as e:
            error_message = str(e)

    return render_template("index.html", result=result, error_message=error_message)


@app.route("/history")
def history():
    rows = get_last_analyses(limit=20)
    # rows: (id, created_at, source_type, source_input,
    #        summary, sentiment_label, sentiment_score,
    #        bias_score, bias_label)
    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
