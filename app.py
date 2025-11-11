import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from newspaper import Article
from langdetect import detect, LangDetectException

app = Flask(__name__)

DB_PATH = "news.db"

# -------------------------------------------------
# 1) VERİTABANI YARDIMCI FONKSİYONLARI
# -------------------------------------------------

def init_db():
    """
    İlk tabloyu oluşturur.
    Eski versiyondan kalan news.db varsa, bunu silersen tablo sıfırdan oluşturulur.
    """
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
            source_input[:200],
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


def get_analysis_by_id(analysis_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT id, created_at, source_type, source_input,
               summary, sentiment_label, sentiment_score,
               bias_score, bias_label
        FROM analyses
        WHERE id = ?
        """,
        (analysis_id,),
    )
    row = c.fetchone()
    conn.close()
    return row


# -------------------------------------------------
# 2) HUGGINGFACE MODELLERİ
# -------------------------------------------------

# Özetleme için BART (İngilizce odaklı, ama demoda iş görür)
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# Duygu analizi modeli
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)


# -------------------------------------------------
# 3) HABER METNİNİ URL'DEN ÇEKME
# -------------------------------------------------

def get_article_text_from_url(url: str) -> str:
    article = Article(url)
    article.download()
    article.parse()
    return article.text


# -------------------------------------------------
# 4) BIAS SKORU HESAPLAMA (GELİŞTİRİLMİŞ HEURİSTİK)
# -------------------------------------------------

def compute_bias(sentiment_label: str, sentiment_score: float, text_length: int):
    """
    Basit ama biraz daha mantıklı bir bias hesaplama:

    - Pozitif / Negatif sentiment ise -> baz bias yüksek (70)
    - Nötr ise -> baz bias düşük (20)
    - Metin çok kısaysa -> bias güvenilirlik düşük, skoru biraz aşağı çek
    - Metin çok uzunsa -> sentiment güçlü ise bias artır

    Gerçek hayatta bunu özel modellerle yapmak gerekir;
    bu proje için açıklanabilir bir demo heuristiği.
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
        bias_label = "Göreceli olarak tarafsız / nötr"

    # sentiment_score 0–1 arası: 0.5 üstü -> daha güçlü ton
    bias_score = base + (sentiment_score - 0.5) * 60  # -30 ile +30 arası oynat

    # Metin çok kısa ise (ör. < 500 karakter) -> skoru biraz kırp
    if text_length < 500:
        bias_score *= 0.7
        bias_label += " – kısa metin (düşük güven)"

    # Metin çok uzunsa ve sentiment yüksekse -> biraz artır
    if text_length > 3000 and sentiment_score > 0.7:
        bias_score += 10

    # 0–100 aralığına sıkıştır
    bias_score = int(min(100, max(0, bias_score)))

    return bias_score, bias_label


# -------------------------------------------------
# 5) ANALİZİ TEK BİR FONKSİYONA TOPLAMA
# -------------------------------------------------

def perform_analysis(source_type: str, source_input: str, full_text: str):
    # 1) Dil tespiti
    try:
        language = detect(full_text[:1000])
    except LangDetectException:
        language = "unknown"

    text_length = len(full_text)

    # 2) Özet üretme
    # BART çok uzun inputlarda hata verebilir, o yüzden truncate ediyoruz
    if text_length > 2000:
        summarization_input = full_text[:2000]
    else:
        summarization_input = full_text

    summary_chunks = summarizer(
        summarization_input,
        max_length=150,
        min_length=40,
        do_sample=False
    )
    summary_text = summary_chunks[0]["summary_text"]

    # 3) Sentiment analizi
    sentiment_result = sentiment_model(summary_text)[0]
    sentiment_label = sentiment_result["label"]
    sentiment_score = float(sentiment_result["score"])

    # 4) Bias hesapla
    bias_score, bias_label = compute_bias(sentiment_label, sentiment_score, text_length)

    # 5) DB'ye kaydet
    save_analysis(
        source_type=source_type,
        source_input=source_input,
        summary=summary_text,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        bias_score=bias_score,
        bias_label=bias_label,
    )

    # 6) Frontend / API için döndür
    result = {
        "full_text": full_text,
        "summary": summary_text,
        "sentiment_label": sentiment_label,
        "sentiment_score": round(sentiment_score, 3),
        "bias_score": bias_score,
        "bias_label": bias_label,
        "source_type": source_type,
        "source_input": source_input,
        "language": language,
        "text_length": text_length,
    }
    return result


# -------------------------------------------------
# 6) FLASK ROTALARI – WEB ARAYÜZÜ
# -------------------------------------------------

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

            # 2) Analizi yap
            result = perform_analysis(source_type, source_input, full_text)

        except Exception as e:
            error_message = str(e)

    return render_template("index.html", result=result, error_message=error_message)


@app.route("/history")
def history():
    # İstersek query param ile filtre ekleyebiliriz (ör. sentiment=POSITIVE)
    sentiment_filter = request.args.get("sentiment")
    rows = get_last_analyses(limit=50)

    if sentiment_filter:
        sentiment_filter = sentiment_filter.upper()
        rows = [r for r in rows if sentiment_filter in (r[5] or "").upper()]

    return render_template("history.html", rows=rows, sentiment_filter=sentiment_filter)


@app.route("/analysis/<int:analysis_id>")
def analysis_detail(analysis_id):
    row = get_analysis_by_id(analysis_id)
    if row is None:
        return f"Analysis with id={analysis_id} not found.", 404

    # row: (id, created_at, source_type, source_input,
    #       summary, sentiment_label, sentiment_score,
    #       bias_score, bias_label)
    return render_template("analysis.html", row=row)


# -------------------------------------------------
# 7) FLASK ROTALARI – JSON API (GELİŞMİŞ)
# -------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    JSON body:
    {
        "input_type": "url" | "text",
        "url": "...",
        "raw_text": "..."
    }
    """
    try:
        data = request.get_json(force=True) or {}
        input_type = data.get("input_type")
        url = data.get("url") or ""
        raw_text = data.get("raw_text") or ""

        if input_type == "url" and url.strip():
            full_text = get_article_text_from_url(url.strip())
            source_type = "url"
            source_input = url.strip()
        elif input_type == "text" and raw_text.strip():
            full_text = raw_text.strip()
            source_type = "text"
            source_input = raw_text.strip()
        else:
            return jsonify({"status": "error", "message": "URL veya metin zorunlu."}), 400

        if len(full_text) < 50:
            return jsonify({"status": "error", "message": "Haber metni çok kısa."}), 400

        result = perform_analysis(source_type, source_input, full_text)
        return jsonify({"status": "ok", "result": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    limit = request.args.get("limit", default=20, type=int)
    rows = get_last_analyses(limit=limit)
    data = []
    for r in rows:
        item = {
            "id": r[0],
            "created_at": r[1],
            "source_type": r[2],
            "source_input": r[3],
            "summary": r[4],
            "sentiment_label": r[5],
            "sentiment_score": r[6],
            "bias_score": r[7],
            "bias_label": r[8],
        }
        data.append(item)
    return jsonify({"status": "ok", "items": data})


# -------------------------------------------------
# 8) MAIN
# -------------------------------------------------

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
