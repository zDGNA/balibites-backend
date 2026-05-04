FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch DULU — potong image dari 2.6GB → ~800MB
RUN pip install --no-cache-dir \
    + torch==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file (termasuk .csv dan .npy)
COPY . .

# Bake model ke /app/model saat BUILD — startup tidak perlu download
# Model: firqaaa/indo-sentence-bert-base (~100MB, ringan)
RUN python - << 'PYEOF'
from sentence_transformers import SentenceTransformer
import os

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SAVE_PATH = "/app/model"

print(f"Downloading {MODEL_ID} ...")
m = SentenceTransformer(MODEL_ID)
m.save(SAVE_PATH)

# Verifikasi
files = os.listdir(SAVE_PATH)
print(f"Saved files: {files}")

m2   = SentenceTransformer(SAVE_PATH)
test = m2.encode(["test bali"])
print(f"Verify OK — shape: {test.shape}")
PYEOF

EXPOSE 8000
CMD ["python", "main.py"]
