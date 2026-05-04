FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi
COPY . .

# Railway akan memberikan port secara dinamis, 
# jadi kita tidak perlu EXPOSE angka spesifik, tapi ini opsional sebagai dokumentasi
EXPOSE 8000

# Jalankan server dengan variabel $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]