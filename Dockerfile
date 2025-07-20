# Gunakan image dasar Python
FROM python:3.9-slim-buster

# Setel direktori kerja di dalam container
WORKDIR /app

# Salin requirements.txt dan instal dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek Anda ke dalam container
COPY . .

# Streamlit berjalan di port 8501 secara default
EXPOSE 8501

# Perintah yang dijalankan saat container dimulai
# --server.port=8501 memastikan Streamlit mendengarkan port 8501
# --server.enableCORS=false dan --server.enableXsrfProtection=false penting untuk deployment
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]