"""
================================================================
  BALIBITES — FastAPI Backend (Railway)
  Lokal: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
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

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIG
#
# MODEL_PATH:
#   - JANGAN diisi dengan nama HuggingFace ("indobenchmark/...") di Railway
#     env var — ini yang menyebabkan warning "No ST model found"
#   - Biarkan kosong / hapus dari Railway dashboard
#   - Default: /app/model  (model sudah di-bake di Dockerfile)
#
# CSV_PATH & VECTORS_PATH:
#   - Sudah benar di Railway dashboard: /app/balibites_ready_to_embed.csv
#   - Default fallback juga mengarah ke /app/
# ─────────────────────────────────────────────────────────────────────────────

APP_DIR = os.path.dirname(os.path.abspath(__file__))  # /app di Railway

# Model: pakai env var HANYA kalau isinya path folder yang valid
_env_model = os.getenv("MODEL_PATH", "")
if _env_model and os.path.isdir(_env_model):
    MODEL_PATH = _env_model
else:
    # Abaikan env var jika bukan path folder — cegah nama HuggingFace masuk
    MODEL_PATH = os.path.join(APP_DIR, "model")

VECTORS_PATH = os.getenv("VECTORS_PATH", os.path.join(APP_DIR, "balibites_indobert_vectors.npy"))
CSV_PATH     = os.getenv("CSV_PATH",     os.path.join(APP_DIR, "balibites_ready_to_embed.csv"))

print("=" * 60)
print(f"APP_DIR      : {APP_DIR}")
print(f"MODEL_PATH   : {MODEL_PATH}  | is_dir={os.path.isdir(MODEL_PATH)}")
print(f"VECTORS_PATH : {VECTORS_PATH}  | exists={os.path.exists(VECTORS_PATH)}")
print(f"CSV_PATH     : {CSV_PATH}  | exists={os.path.exists(CSV_PATH)}")
print("=" * 60)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="BaliBites API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model dari /app/model (bukan download) ───────────────────────────────
print("Loading model from disk...")
if not os.path.isdir(MODEL_PATH):
    raise RuntimeError(
        f"Model folder tidak ditemukan: '{MODEL_PATH}'\n"
        "Pastikan Dockerfile sudah RUN python -c 'model.save(/app/model)'"
    )
model = SentenceTransformer(MODEL_PATH)
print(f"Model loaded OK dari {MODEL_PATH}")

# ── Load vectors ──────────────────────────────────────────────────────────────
if not os.path.exists(VECTORS_PATH):
    raise RuntimeError(f"Vectors tidak ditemukan: {VECTORS_PATH}")
vectors = np.load(VECTORS_PATH)
print(f"Vectors loaded: shape {vectors.shape}")

# ── Load CSV ──────────────────────────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    raise RuntimeError(f"CSV tidak ditemukan: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Dataset loaded: {len(df)} restoran")

# ── Slang map ─────────────────────────────────────────────────────────────────
SLANG_MAP = {
    "murah": "budget friendly",        "murmer": "budget friendly",
    "terjangkau": "budget friendly",   "mahal": "premium fine dining",
    "enak": "lezat berkualitas",       "mantap": "lezat berkualitas",
    "josss": "lezat berkualitas",      "hits": "populer instagramable",
    "kekinian": "modern instagramable","aesthetic": "instagramable cozy",
    "cozy": "nyaman cozy",             "santai": "nyaman cozy",
    "romantis": "romantic intimate",
    "babi guling": "babi guling kuliner khas bali pork",
    "be guling":   "babi guling kuliner khas bali pork",
    "ayam betutu": "ayam betutu kuliner khas bali chicken",
    "sate lilit":  "sate lilit kuliner khas bali satay",
    "nasi campur": "nasi campur kuliner khas bali mixed rice",
    "sarapan": "breakfast pagi", "makan siang": "lunch siang", "makan malam": "dinner malam",
}

REKOMENDASI_KW = {
    "cari","cariin","rekomendasiin","rekomendasikan","rekomendasi",
    "suggest","sarankan","saran","dimana","mau makan","pengen makan",
    "mau coba","cobain","ada tempat","tempat makan","restoran apa",
    "yang enak","yang hits","yang bagus",
}
DETAIL_KW = {
    "jam buka","jam tutup","berapa","harga","menu","kontak",
    "lokasi","alamat","review","rating","ulasan","promo",
    "diskon","gofood","grabfood","bisa delivery",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
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

def clean(val, fallback="N/A") -> str:
    return str(val).strip() if pd.notnull(val) and str(val).strip() not in ("", "nan") else fallback

def parse_highlights(val) -> list:
    raw = clean(val)
    if raw == "N/A":
        return []
    try:
        import json
        return json.loads(raw)
    except Exception:
        return [h.strip() for h in raw.split(",") if h.strip()]

def row_to_dict(row, rank=0, final_score=0.0, sim_score=0.0) -> dict:
    return {
        "id":            int(row.name),
        "rank":          rank,
        "nama":          clean(row["nama"]),
        "kabupaten":     clean(row["kabupaten"]),
        "kategori":      clean(row["kategori"]),
        "rating":        float(row["rating"])      if pd.notnull(row.get("rating"))      else 0.0,
        "total_review":  int(row["total_review"])  if pd.notnull(row.get("total_review")) else 0,
        "price_range":   clean(row.get("price_range")),
        "best_time":     clean(row.get("best_time")),
        "top_menu":      clean(row.get("Top Menu")),
        "promo":         clean(row.get("Promo")),
        "highlights":    parse_highlights(row.get("highlights")),
        "link":          clean(row.get("link")),
        "gofood":        clean(row.get("GoFood_Link")),
        "grabfood":      clean(row.get("GrabFood_Link")),
        "tiktok":        clean(row.get("tiktok_link_1")),
        "yt":            clean(row.get("yt_link_1")),
        "reels":         clean(row.get("reels_link_1")),
        "lat":           float(row["lat"]) if pd.notnull(row.get("lat")) else None,
        "lng":           float(row["lng"]) if pd.notnull(row.get("lng")) else None,
        "score_final":   round(float(final_score), 4),
        "score_semantic":round(float(sim_score),   4),
    }

def detect_intent(text: str) -> dict:
    lower = text.lower()
    words = set(re.sub(r"[^\w\s]", " ", lower).split())
    matched = None
    for nama in df["nama"].dropna().str.lower():
        first = nama.split()[0]
        if len(first) > 3 and first in lower:
            matched = nama
            break
    sr = len(words & REKOMENDASI_KW)
    sd = len(words & DETAIL_KW) + (3 if matched else 0)
    if sr > sd:
        return {"intent": "REKOMENDASI", "matched_restaurant": None}
    elif sd > 0:
        return {"intent": "DETAIL", "matched_restaurant": matched}
    return {"intent": "REKOMENDASI", "matched_restaurant": None}

# ── Schemas ───────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    query:      str
    lokasi:     Optional[str]   = ""
    top_n:      Optional[int]   = 5
    w_semantic: Optional[float] = 0.60
    w_rating:   Optional[float] = 0.25
    w_review:   Optional[float] = 0.15

class ChatRequest(BaseModel):
    message: str
    lokasi:  Optional[str] = ""

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "restaurants": len(df), "vectors": int(vectors.shape[0])}


@app.get("/restaurant")
def get_restaurants(
    min_rating: float = 4.0,
    min_review: int   = 30,
    top_n:      int   = 40,
    kabupaten:  str   = "",
    kategori:   str   = "",
):
    """Load marker map — dipakai oleh Next.js ChatPanel & BaliMap."""
    mask = pd.Series([True] * len(df))
    if kabupaten:
        mask &= df["kabupaten"].str.contains(kabupaten, case=False, na=False)
    if kategori:
        mask &= df["kategori"].str.contains(kategori, case=False, na=False)
    mask &= df["rating"].fillna(0)      >= min_rating
    mask &= df["total_review"].fillna(0) >= min_review

    filtered = df[mask].sort_values("rating", ascending=False).head(top_n)
    return [row_to_dict(row) for _, row in filtered.iterrows()
            if pd.notnull(row.get("lat")) and pd.notnull(row.get("lng"))]


@app.post("/recommend")
def recommend(req: RecommendRequest):
    mask = (
        df["kabupaten"].str.contains(req.lokasi, case=False, na=False)
        if req.lokasi else pd.Series([True] * len(df))
    )
    df_loc      = df[mask].reset_index(drop=True)
    idx_loc     = df.index[mask].tolist()
    vectors_loc = vectors[idx_loc]
    if df_loc.empty:
        raise HTTPException(404, f"Tidak ada data untuk lokasi: {req.lokasi}")

    q_vec      = model.encode([preprocess(req.query)], normalize_embeddings=True)
    sim_scores = (q_vec @ vectors_loc.T).flatten()
    ratings    = df_loc["rating"].fillna(0).values.astype(float)
    reviews    = df_loc["total_review"].fillna(0).values.astype(float)

    final = (
        req.w_semantic * normalize(sim_scores) +
        req.w_rating   * normalize(ratings) +
        req.w_review   * normalize(np.log1p(reviews))
    )
    top_idx = final.argsort()[-req.top_n:][::-1]
    results = [row_to_dict(df_loc.iloc[i], rank=r, final_score=final[i], sim_score=sim_scores[i])
               for r, i in enumerate(top_idx, 1)]

    return {"query": req.query, "lokasi": req.lokasi, "total_found": len(df_loc), "results": results}


@app.get("/detail/{nama}")
def detail(nama: str):
    mask = df["nama"].str.lower() == nama.lower()
    if mask.sum() == 0:
        raise HTTPException(404, f"'{nama}' tidak ditemukan")
    return row_to_dict(df[mask].iloc[0])


@app.post("/chat")
def chat(req: ChatRequest):
    ir = detect_intent(req.message)
    if ir["intent"] == "DETAIL" and ir["matched_restaurant"]:
        mask = df["nama"].str.lower() == ir["matched_restaurant"].lower()
        if mask.sum() > 0:
            return {"intent": "DETAIL", "restaurant": row_to_dict(df[mask].iloc[0])}
    rec = recommend(RecommendRequest(query=req.message, lokasi=req.lokasi, top_n=5))
    return {"intent": "REKOMENDASI", **rec}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
