"""
================================================================
  BALIBITES — FastAPI Backend (Accuracy Optimized)
  Lokal: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
================================================================
"""

import re, os, json
import numpy as np
import pandas as pd
import math
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Optional

# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Hitung jarak antara 2 titik koordinat dalam Kilometer"""
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ── Paths & Config ────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))

_env_model   = os.getenv("MODEL_PATH", "")
MODEL_PATH   = _env_model if (_env_model and os.path.isdir(_env_model)) else os.path.join(APP_DIR, "model")
VECTORS_PATH = os.getenv("VECTORS_PATH", os.path.join(APP_DIR, "balibites_indobert_vectors.npy"))
CSV_PATH     = os.getenv("CSV_PATH",     os.path.join(APP_DIR, "balibites_ready_to_embed.csv"))
MODEL_ID     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Daftar area Bali untuk ekstraksi lokasi otomatis
BALI_AREAS = [
    "canggu", "berawa", "pererenan", "echo beach", "batu bolong",
    "ubud", "gianyar", "payangan", "tegallalang",
    "seminyak", "legian", "kuta", "kerobokan",
    "sanur", "denpasar", "renon",
    "uluwatu", "pecatu", "bingin", "padang padang", "dreamland",
    "jimbaran", "nusa dua", "tanjung benoa",
    "tabanan", "tanah lot", "kediri",
    "bangli", "kintamani", "penelokan",
    "karangasem", "amlapura", "candidasa", "tirta gangga",
    "buleleng", "lovina", "singaraja", "munduk"
]

print("=" * 60)
print(f"MODEL_PATH   : {MODEL_PATH}  | is_dir={os.path.isdir(MODEL_PATH)}")
print(f"VECTORS_PATH : {VECTORS_PATH}  | exists={os.path.exists(VECTORS_PATH)}")
print(f"CSV_PATH     : {CSV_PATH}  | exists={os.path.exists(CSV_PATH)}")
print("=" * 60)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="BaliBites API", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Load Model ────────────────────────────────────────────────────────────────
def _has_weights(path: str) -> bool:
    if not os.path.isdir(path): return False
    return any(f in os.listdir(path) for f in {"model.safetensors", "pytorch_model.bin"})

print("Loading model...")
if _has_weights(MODEL_PATH):
    model = SentenceTransformer(MODEL_PATH)
    print(f"Loaded from disk: {MODEL_PATH}")
else:
    print(f"Downloading {MODEL_ID}...")
    model = SentenceTransformer(MODEL_ID)
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")

MODEL_DIM = model.get_sentence_embedding_dimension()
print(f"Model dim: {MODEL_DIM}")

# ── Load Data ─────────────────────────────────────────────────────────────────
if not os.path.exists(VECTORS_PATH):
    raise RuntimeError(f"Vectors tidak ditemukan: {VECTORS_PATH}")
vectors = np.load(VECTORS_PATH)
print(f"Vectors shape: {vectors.shape}")

VECTOR_DIM = vectors.shape[1]
DIM_OK = (MODEL_DIM == VECTOR_DIM)
if not DIM_OK:
    print(f"⚠️  DIMENSION MISMATCH: model={MODEL_DIM}, vectors={VECTOR_DIM}")
else:
    print(f"✅ Dimension match: {MODEL_DIM}")

if not os.path.exists(CSV_PATH):
    raise RuntimeError(f"CSV tidak ditemukan: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Sinkronisasi
if len(df) != vectors.shape[0]:
    print(f"⚠️  Row mismatch df={len(df)} vectors={vectors.shape[0]}, truncating...")
    min_len = min(len(df), vectors.shape[0])
    df      = df.iloc[:min_len].reset_index(drop=True)
    vectors = vectors[:min_len]

print(f"Dataset: {len(df)} restoran loaded.\n")

# ── Slang Map ─────────────────────────────────────────────────────────────────
SLANG_MAP = {
    "murah":"budget friendly","murmer":"budget friendly","terjangkau":"budget friendly",
    "mahal":"premium fine dining","enak":"lezat berkualitas","mantap":"lezat berkualitas",
    "josss":"lezat berkualitas","hits":"populer instagramable","kekinian":"modern instagramable",
    "aesthetic":"instagramable cozy","cozy":"nyaman cozy","santai":"nyaman cozy",
    "romantis":"romantic intimate",
    "babi guling":"babi guling kuliner khas bali pork",
    "be guling":"babi guling kuliner khas bali pork",
    "ayam betutu":"ayam betutu kuliner khas bali chicken",
    "sate lilit":"sate lilit kuliner khas bali satay",
    "nasi campur":"nasi campur kuliner khas bali mixed rice",
    "sarapan":"breakfast pagi","makan siang":"lunch siang","makan malam":"dinner malam",
}

# ── Core Functions ────────────────────────────────────────────────────────────

def extract_location(text: str) -> str:
    """🔥 FIX: Deteksi nama daerah Bali dari query text"""
    text_lower = text.lower()
    for area in BALI_AREAS:
        if area in text_lower:
            return area
    return ""

def preprocess(text: str) -> str:
    text = text.lower().strip()
    for s, r in SLANG_MAP.items():
        text = re.sub(r'\b' + re.escape(s) + r'\b', r, text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return np.zeros_like(arr, dtype=float) if hi - lo < 1e-9 else (arr - lo) / (hi - lo)

def clean(val, fallback="N/A") -> str:
    v = str(val).strip() if pd.notnull(val) else ""
    return v if v and v != "nan" else fallback

def parse_highlights(val) -> list:
    raw = clean(val)
    if raw == "N/A": return []
    try:
        parsed = json.loads(raw)
        return [str(h).strip() for h in (parsed if isinstance(parsed, list) else [parsed]) if h]
    except Exception:
        return [h.strip() for h in raw.split(",") if h.strip() and h.strip() != "System.Object[]"]

def row_to_dict(row, rank=0, final_score=0.0, sim_score=0.0) -> dict:
    return {
        "id":            int(row.name),
        "rank":          rank,
        "nama":          clean(row["nama"]),
        "kabupaten":     clean(row["kabupaten"]),
        "kategori":      clean(row["kategori"]),
        "rating":        float(row["rating"])     if pd.notnull(row.get("rating"))      else 0.0,
        "total_review":  int(row["total_review"]) if pd.notnull(row.get("total_review")) else 0,
        "price_range":   clean(row.get("price_range")),
        "best_time":     clean(row.get("best_time")),
        "top_menu":      clean(row.get("Top Menu")),
        "promo":         clean(row.get("Promo")),
        "highlights":    parse_highlights(row.get("highlights")),
        "link":          clean(row.get("link")),
        "gofood":        clean(row.get("GoFood_Link")),
        "grabfood":      clean(row.get("GrabFood_Link")),
        "tiktok1":        clean(row.get("tiktok_link_1")),
        "tiktok2":        clean(row.get("tiktok_link_2")),
        "tiktok3":        clean(row.get("tiktok_link_3")),
        "yt1":            clean(row.get("yt_link_1")),
        "yt2":            clean(row.get("yt_link_2")),
        "yt3":            clean(row.get("yt_link_3")),
        "reels1":         clean(row.get("reels_link_1")),
        "reels2":         clean(row.get("reels_link_2")),
        "reels3":         clean(row.get("reels_link_3")),
        "lat":           float(row["lat"]) if pd.notnull(row.get("lat")) else None,
        "lng":           float(row["lng"]) if pd.notnull(row.get("lng")) else None,
        "score_final":   round(float(final_score), 4),
        "score_semantic":round(float(sim_score),   4),
    }

def vector_search(query: str, df_sub: pd.DataFrame, vecs: np.ndarray,
                  top_n: int = 5, w_sem=0.70, w_rat=0.20, w_rev=0.10) -> list:
    """
    🔥 FIX: Semantic search dengan Threshold & Bobot Akurasi Tinggi
    """
    if not DIM_OK: return []
    
    q_vec = model.encode([preprocess(query)], normalize_embeddings=True)
    sim_scores = (q_vec @ vecs.T).flatten()
    
    # 🔒 Threshold: Abaikan hasil yang similarity-nya terlalu rendah (< 0.30)
    MIN_SIM = 0.30
    valid_mask = sim_scores >= MIN_SIM
    
    if not valid_mask.any():
        return []

    # Filter data yang valid saja
    valid_indices = np.where(valid_mask)[0]
    valid_sims = sim_scores[valid_mask]
    valid_df = df_sub.iloc[valid_indices].reset_index(drop=True)
    
    # Hitung skor final
    ratings = valid_df["rating"].fillna(0).values.astype(float)
    reviews = valid_df["total_review"].fillna(0).values.astype(float)
    
    final = w_sem * normalize(valid_sims) + w_rat * normalize(ratings) + w_rev * normalize(np.log1p(reviews))
    
    # Sort dan ambil top N
    top_local_idx = final.argsort()[-top_n:][::-1]
    
    results = []
    for local_idx in top_local_idx:
        row = valid_df.iloc[local_idx]
        # Ambil skor semantic asli dari array
        # Mapping index lokal ke index di valid_sims
        score = valid_sims[local_idx] 
        results.append(row_to_dict(row, rank=len(results)+1, final_score=final[local_idx], sim_score=score))
        
    return results

def keyword_search(query: str, df_sub: pd.DataFrame, top_n: int = 5) -> list:
    kws = set(preprocess(query).split())
    scored = []
    for idx, row in df_sub.iterrows():
        text = f"{row.get('nama','')} {row.get('kategori','')} {row.get('kabupaten','')}".lower()
        hits = sum(1 for k in kws if k in text)
        if hits:
            scored.append((idx, hits + float(row.get("rating", 0)) / 5))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [row_to_dict(df_sub.loc[i], rank=r+1) for r, (i, _) in enumerate(scored[:top_n])]

def detect_intent(text: str) -> dict:
    lower = text.lower()
    matched = None
    for nama in df["nama"].dropna().str.lower():
        if isinstance(nama, str) and len(nama) >= 4 and nama in lower:
            matched = nama
            break

    detail_kw = {"cerita","detail","tentang","worth it","tips","review","ulasan",
                 "bagaimana","recommended","worth","bagus","jam buka","harga","menu",
                 "kontak","alamat","promo","gofood","grabfood"}
    rekom_kw  = {" di ","dekat","area","murah","mahal","budget","terbaik","populer",
                 "hits","cozy","romantis","family","wifi","sunset","beach","view",
                 "cafe","warung","restoran","tempat makan","kuliner","sarapan"}

    has_detail = any(k in lower for k in detail_kw)
    has_rekom  = any(k in lower for k in rekom_kw)

    if matched and has_detail:
        return {"intent": "DETAIL", "matched": matched}
    if has_rekom or not matched:
        return {"intent": "REKOMENDASI", "matched": None}
    if matched:
        return {"intent": "DETAIL", "matched": matched}
    return {"intent": "REKOMENDASI", "matched": None}

# ── Schemas ───────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    query:      str
    lokasi:     Optional[str]   = ""
    top_n:      Optional[int]   = 10
    w_semantic: Optional[float] = 0.70
    w_rating:   Optional[float] = 0.20
    w_review:   Optional[float] = 0.10

class ChatRequest(BaseModel):
    message:       str
    lokasi:        Optional[str]  = ""
    context:       Optional[dict] = None
    user_location: Optional[dict] = None
    preferences:   Optional[dict] = None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok", "restaurants": len(df), "vectors": int(vectors.shape[0]),
        "model_dim": MODEL_DIM, "vec_dim": VECTOR_DIM, "dim_ok": DIM_OK,
    }

@app.get("/restaurant")
def get_restaurants(
    min_rating: float = 3.0, min_review: int = 30, top_n: int = 40,
    kabupaten: str = "", kategori: str = "",
    user_lat: Optional[float] = None, user_lng: Optional[float] = None,
    sort_by: str = "rating"
):
    mask = pd.Series([True] * len(df))
    if kabupaten: mask &= df["kabupaten"].str.contains(kabupaten, case=False, na=False)
    if kategori:  mask &= df["kategori"].str.contains(kategori, case=False, na=False)
    mask &= df["rating"].fillna(0) >= min_rating
    mask &= df["total_review"].fillna(0) >= min_review

    filtered = df[mask].copy()
    
    if user_lat is not None and user_lng is not None and not filtered.empty:
        filtered.loc[:, "distance_km"] = filtered.apply(
            lambda row: haversine(user_lat, user_lng, row["lat"], row["lng"]), axis=1)
        if sort_by == "distance":
            filtered = filtered.sort_values("distance_km", ascending=True)
        else:
            filtered = filtered.sort_values("rating", ascending=False)
    else:
        filtered = filtered.sort_values("rating", ascending=False)

    results = []
    for _, row in filtered.head(top_n).iterrows():
        resto = row_to_dict(row)
        if "distance_km" in row and pd.notnull(row["distance_km"]):
            resto["distance_km"] = round(row["distance_km"], 2)
        results.append(resto)
    return results

@app.post("/recommend")
def recommend(req: RecommendRequest):
    # 🔥 FIX: Extract location if not provided
    lokasi = req.lokasi or extract_location(req.query)
    
    # 🔥 FIX: Broad filter (Kabupaten OR Nama OR Kategori)
    mask = pd.Series([False] * len(df))
    if lokasi:
        loc_l = lokasi.lower()
        mask = (df["kabupaten"].str.contains(loc_l, case=False, na=False) |
                df["nama"].str.contains(loc_l, case=False, na=False) |
                df["kategori"].str.contains(loc_l, case=False, na=False))
    else:
        mask = pd.Series([True] * len(df))
        
    df_loc = df[mask].reset_index(drop=True)
    idx_loc = df.index[mask].tolist()
    vectors_loc = vectors[idx_loc]
    
    if df_loc.empty:
        raise HTTPException(404, f"Tidak ada data untuk lokasi: {lokasi}")

    if DIM_OK:
        results = vector_search(req.query, df_loc, vectors_loc, req.top_n, req.w_semantic, req.w_rating, req.w_review)
    else:
        results = keyword_search(req.query, df_loc, req.top_n)

    return {"query": req.query, "lokasi": lokasi, "total_found": len(df_loc), "results": results}

@app.get("/detail/{nama}")
def detail(nama: str):
    import urllib.parse
    nama_decoded = urllib.parse.unquote(nama)
    
    # Case-insensitive match
    mask = df["nama"].str.lower() == nama_decoded.lower()
    
    if mask.sum() == 0:
        # Return JSON error, bukan plain text!
        raise HTTPException(
            status_code=404, 
            detail=f"'{nama_decoded}' tidak ditemukan. Coba cek ejaan."
        )
    
    return row_to_dict(df[mask].iloc[0])

@app.post("/chat")
def chat(req: ChatRequest):
    ir = detect_intent(req.message)

    # ── DETAIL ────────────────────────────────────────────────────────────────
    if ir["intent"] == "DETAIL" and ir["matched"]:
        mask = df["nama"].str.lower() == ir["matched"].lower()
        if mask.sum() > 0:
            resto = row_to_dict(df[mask].iloc[0], rank=1)
            return {
                "intent": "DETAIL",
                "reply": f"Ini detail lengkap **{resto['nama']}** 🍽️",
                "results": None,
                "restaurant": resto,
            }

    # ── REKOMENDASI ───────────────────────────────────────────────────────────
    # 🔥 FIX: Extract location from message if missing
    lokasi = req.lokasi or (req.context or {}).get("kabupaten", "") or extract_location(req.message)
    
    # 🔥 FIX: Broad filter
    mask = pd.Series([False] * len(df))
    if lokasi:
        loc_l = lokasi.lower()
        mask = (df["kabupaten"].str.contains(loc_l, case=False, na=False) |
                df["nama"].str.contains(loc_l, case=False, na=False) |
                df["kategori"].str.contains(loc_l, case=False, na=False))
    else:
        mask = pd.Series([True] * len(df))
        
    df_loc = df[mask].reset_index(drop=True)
    idx_loc = df.index[mask].tolist()
    vectors_loc = vectors[idx_loc]

    if df_loc.empty:
        return {
            "intent": "REKOMENDASI",
            "reply": f"Maaf, tidak ada data untuk area '{lokasi}'.",
            "results": [], "restaurant": None,
        }

    if DIM_OK:
        results = vector_search(req.message, df_loc, vectors_loc, top_n=5)
        method = "semantic"
    else:
        results = keyword_search(req.message, df_loc, top_n=5)
        method = "keyword"

    if not results:
        return {
            "intent": "REKOMENDASI",
            "reply": f"Tidak ada hasil cocok untuk \"{req.message}\".",
            "results": [], "restaurant": None,
        }

    return {
        "intent": "REKOMENDASI",
        "reply": f"Saya temukan **{len(results)}** rekomendasi untuk \"{req.message}\" 🍽️",
        "results": results,
        "restaurant": None,
        "_method": method,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))