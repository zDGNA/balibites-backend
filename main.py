"""
================================================================
  BALIBITES — FastAPI Backend
  File: main.py
  Jalankan: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
================================================================
"""

import re
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Optional

# ── Config ────────────────────────────────────────────────────
# Mengambil path folder 'backend' tempat main.py berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sesuaikan dengan nama file hasil generate di Notebook tadi
VECTORS_PATH = os.path.join(BASE_DIR, "balibites_indobert_vectors.npy")
CSV_PATH     = os.path.join(BASE_DIR, "balibites_processed.csv")

# Model path tetap menggunakan HuggingFace (online download sekali)
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="BaliBites API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ganti dengan domain Next.js Anda saat production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model & data saat startup (sekali) ───────────────────
print("🔄 Memuat model dan data BaliBites...")

# Memastikan model menggunakan nama yang konsisten
model = SentenceTransformer(MODEL_NAME)

# Load data lokal
if os.path.exists(VECTORS_PATH) and os.path.exists(CSV_PATH):
    vectors = np.load(VECTORS_PATH)
    df      = pd.read_csv(CSV_PATH)
    print(f"✅ Siap! {len(df)} restoran dan {vectors.shape[0]} vektor dimuat.")
else:
    print(f"❌ ERROR: File data tidak ditemukan!")
    print(f"Pastikan file ada di:\n- {CSV_PATH}\n- {VECTORS_PATH}")

# ── Kamus Slang ───────────────────────────────────────────────
SLANG_MAP = {
    "murah": "budget friendly", "murmer": "budget friendly",
    "terjangkau": "budget friendly", "mahal": "premium fine dining",
    "enak": "lezat berkualitas", "mantap": "lezat berkualitas",
    "josss": "lezat berkualitas", "hits": "populer instagramable",
    "kekinian": "modern instagramable", "aesthetic": "instagramable cozy",
    "cozy": "nyaman cozy", "santai": "nyaman cozy",
    "romantis": "romantic intimate",
    "babi guling": "babi guling kuliner khas bali pork",
    "be guling": "babi guling kuliner khas bali pork",
    "ayam betutu": "ayam betutu kuliner khas bali chicken",
    "sate lilit": "sate lilit kuliner khas bali satay",
    "nasi campur": "nasi campur kuliner khas bali mixed rice",
    "sarapan": "breakfast pagi", "makan siang": "lunch siang",
    "makan malam": "dinner malam",
}

REKOMENDASI_KEYWORDS = {
    "cari", "cariin", "rekomendasiin", "rekomendasikan", "rekomendasi",
    "suggest", "sarankan", "saran", "dimana", "mau makan", "pengen makan",
    "mau coba", "cobain", "ada tempat", "tempat makan", "restoran apa",
    "yang enak", "yang hits", "yang bagus",
}

DETAIL_KEYWORDS = {
    "jam buka", "jam tutup", "berapa", "harga", "menu", "kontak",
    "lokasi", "alamat", "review", "rating", "ulasan", "promo",
    "diskon", "gofood", "grabfood", "bisa delivery",
}

# ── Helpers ───────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower().strip()
    for slang, rep in SLANG_MAP.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', rep, text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)

def clean(val, fallback="N/A"):
    return str(val).strip() if pd.notnull(val) and str(val).strip() not in ("", "nan") else fallback

def detect_intent(text: str) -> dict:
    lower = text.lower()
    words = set(re.sub(r"[^\w\s]", " ", lower).split())

    matched_restaurant = None
    for nama in df["nama"].dropna().str.lower():
        first = nama.split()[0]
        if len(first) > 3 and first in lower:
            matched_restaurant = nama
            break

    score_rek = len(words & REKOMENDASI_KEYWORDS)
    score_det = len(words & DETAIL_KEYWORDS)
    if matched_restaurant:
        score_det += 3

    if score_rek > score_det:
        return {"intent": "REKOMENDASI", "matched_restaurant": None}
    elif score_det > 0:
        return {"intent": "DETAIL", "matched_restaurant": matched_restaurant}
    return {"intent": "REKOMENDASI", "matched_restaurant": None}

# ── Schemas ───────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    query: str
    lokasi: Optional[str] = ""
    top_n: Optional[int] = 5
    w_semantic: Optional[float] = 0.60
    w_rating:   Optional[float] = 0.25
    w_review:   Optional[float] = 0.15

class ChatRequest(BaseModel):
    message: str
    lokasi:  Optional[str] = ""

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"} 

@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Cari rekomendasi restoran berdasarkan query semantik.
    Dipakai langsung oleh Next.js atau melalui /chat.
    """
    mask = (
        df["kabupaten"].str.contains(req.lokasi, case=False, na=False)
        if req.lokasi else pd.Series([True] * len(df))
    )
    df_loc      = df[mask].reset_index(drop=True)
    idx_loc     = df.index[mask].tolist()
    vectors_loc = vectors[idx_loc]

    if df_loc.empty:
        raise HTTPException(404, f"Tidak ada data untuk lokasi: {req.lokasi}")

    q_clean   = preprocess(req.query)
    q_vec     = model.encode([q_clean], normalize_embeddings=True)
    sim_scores = (q_vec @ vectors_loc.T).flatten()

    ratings = df_loc["rating"].fillna(0).values.astype(float)
    reviews = df_loc["total_review"].fillna(0).values.astype(float)

    final = (
        req.w_semantic * normalize(sim_scores) +
        req.w_rating   * normalize(ratings) +
        req.w_review   * normalize(np.log1p(reviews))
    )

    top_idx = final.argsort()[-req.top_n:][::-1]

    results = []
    for rank, i in enumerate(top_idx, start=1):
        row = df_loc.iloc[i]
        results.append({
            "rank":          rank,
            "nama":          clean(row["nama"]),
            "kabupaten":     clean(row["kabupaten"]),
            "kategori":      clean(row["kategori"]),
            "rating":        float(row["rating"]) if pd.notnull(row.get("rating")) else 0.0,
            "total_review":  int(row["total_review"]) if pd.notnull(row.get("total_review")) else 0,
            "price_range":   clean(row.get("price_range")),
            "best_time":     clean(row.get("best_time")),
            "top_menu":      clean(row.get("Top Menu")),
            "promo":         clean(row.get("Promo")),
            "highlights":    clean(row.get("highlights")),
            "gmaps":         clean(row.get("link")),
            "gofood":        clean(row.get("GoFood_Link")),
            "grabfood":      clean(row.get("GrabFood_Link")),
            "tiktok":        clean(row.get("tiktok_link_1")),
            "yt":            clean(row.get("yt_link_1")),
            "reels":         clean(row.get("reels_link_1")),
            "lat":           float(row["lat"]) if pd.notnull(row.get("lat")) else None,
            "lng":           float(row["lng"]) if pd.notnull(row.get("lng")) else None,
            "score_final":   round(float(final[i]), 4),
            "score_semantic":round(float(sim_scores[i]), 4),
        })

    return {
        "query":       req.query,
        "query_clean": q_clean,
        "lokasi":      req.lokasi,
        "total_found": len(df_loc),
        "results":     results,
    }


@app.get("/detail/{nama}")
def detail(nama: str):
    """
    Ambil detail lengkap satu restoran berdasarkan nama.
    """
    mask = df["nama"].str.lower() == nama.lower()
    if mask.sum() == 0:
        raise HTTPException(404, f"Restoran '{nama}' tidak ditemukan")
    row = df[mask].iloc[0]
    return {
        "nama":         clean(row["nama"]),
        "kabupaten":    clean(row["kabupaten"]),
        "kategori":     clean(row["kategori"]),
        "rating":       float(row["rating"]) if pd.notnull(row.get("rating")) else 0,
        "total_review": int(row["total_review"]) if pd.notnull(row.get("total_review")) else 0,
        "price_range":  clean(row.get("price_range")),
        "best_time":    clean(row.get("best_time")),
        "top_menu":     clean(row.get("Top Menu")),
        "promo":        clean(row.get("Promo")),
        "highlights":   clean(row.get("highlights")),
        "gmaps":        clean(row.get("link")),
        "gofood":       clean(row.get("GoFood_Link")),
        "grabfood":     clean(row.get("GrabFood_Link")),
        "tiktok":       clean(row.get("tiktok_link_1")),
        "yt":           clean(row.get("yt_link_1")),
        "reels":        clean(row.get("reels_link_1")),
        "lat":          float(row["lat"]) if pd.notnull(row.get("lat")) else None,
        "lng":          float(row["lng"]) if pd.notnull(row.get("lng")) else None,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Endpoint utama chatbot — deteksi intent lalu routing otomatis.
    Next.js hanya perlu memanggil satu endpoint ini.
    """
    intent_result = detect_intent(req.message)

    if intent_result["intent"] == "DETAIL" and intent_result["matched_restaurant"]:
        nama = intent_result["matched_restaurant"]
        mask = df["nama"].str.lower() == nama.lower()
        if mask.sum() > 0:
            row = df[mask].iloc[0]
            return {
                "intent": "DETAIL",
                "restaurant": {
                    "nama":        clean(row["nama"]),
                    "rating":      float(row["rating"]) if pd.notnull(row.get("rating")) else 0,
                    "total_review":int(row["total_review"]) if pd.notnull(row.get("total_review")) else 0,
                    "kategori":    clean(row["kategori"]),
                    "kabupaten":   clean(row["kabupaten"]),
                    "price_range": clean(row.get("price_range")),
                    "best_time":   clean(row.get("best_time")),
                    "top_menu":    clean(row.get("Top Menu")),
                    "promo":       clean(row.get("Promo")),
                    "gmaps":       clean(row.get("link")),
                    "gofood":      clean(row.get("GoFood_Link")),
                    "grabfood":    clean(row.get("GrabFood_Link")),
                    "tiktok":      clean(row.get("tiktok_link_1")),
                    "lat":         float(row["lat"]) if pd.notnull(row.get("lat")) else None,
                    "lng":         float(row["lng"]) if pd.notnull(row.get("lng")) else None,
                },
            }

    # Default: REKOMENDASI
    rec = recommend(RecommendRequest(
        query=req.message,
        lokasi=req.lokasi,
        top_n=5
    ))
    return {"intent": "REKOMENDASI", **rec}


if __name__ == "__main__":
    import uvicorn
    # Ambil port dari environment Railway, default ke 8000 jika lokal
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)