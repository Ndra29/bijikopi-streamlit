import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2 # Digunakan untuk konversi warna jika diperlukan oleh model

# --- Konfigurasi Model ---
# Pastikan file best.pt ada di folder yang sama dengan app.py ini
MODEL_PATH = "best.pt"

# --- Fungsi untuk Memuat Model ---
# Menggunakan st.cache_resource agar model hanya dimuat sekali
@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari {path}. Pastikan file 'best.pt' ada di folder yang sama. Error: {e}")
        st.stop() # Menghentikan aplikasi jika model tidak bisa dimuat

model = load_yolo_model(MODEL_PATH)

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Klasifikasi Biji Kopi dengan YOLOv8",
    page_icon="☕",
    layout="centered", # Atau "wide" jika ingin layout lebih lebar
    initial_sidebar_state="auto"
)

# --- Judul Aplikasi ---
st.title("☕ Klasifikasi Biji Kopi dengan YOLOv8")
st.write("Unggah gambar biji kopi untuk melihat hasil deteksi dan klasifikasinya.")

# --- Komponen Upload Gambar ---
uploaded_file = st.file_uploader("Upload gambar biji kopi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Tampilkan Gambar Asli ---
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
    st.write("") # Spasi

    st.info("Memproses gambar, mohon tunggu...")

    # --- Prediksi dengan YOLOv8 ---
    # Konversi gambar PIL ke numpy array (RGB)
    image_np = np.array(image)
    # Konversi RGB ke BGR jika model YOLOv8 Anda mengharapkan BGR (umumnya)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Lakukan prediksi. save=False agar tidak menyimpan gambar hasil ke disk
    results = model.predict(source=image_bgr, save=False, conf=0.25, iou=0.7) # conf dan iou bisa disesuaikan

    # --- Tampilkan Hasil Prediksi ---
    st.subheader("Hasil Deteksi:")
    predictions_found = False
    detected_objects_info = []

    for r in results:
        # r.plot() akan menggambar bounding box dan label pada gambar
        # Ini mengembalikan numpy array (BGR), jadi perlu diubah ke RGB untuk st.image
        annotated_image_bgr = r.plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

        st.image(annotated_image_rgb, caption="Hasil Deteksi", use_column_width=True)

        # Ekstrak informasi deteksi (kelas dan confidence)
        if len(r.boxes) > 0:
            for box in r.boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                class_name = model.names.get(class_id, f"Class {class_id}") # Ambil nama kelas dari model
                detected_objects_info.append(f"**{class_name}** (Kepercayaan: `{conf:.2f}`)")
                predictions_found = True
        else:
            detected_objects_info.append("Tidak ada objek terdeteksi dengan ambang batas kepercayaan saat ini.")

    if predictions_found:
        for info in detected_objects_info:
            st.success(info)
    else:
        st.warning("Tidak ada biji kopi yang terdeteksi dalam gambar ini.")
        st.info("Coba unggah gambar lain atau sesuaikan ambang batas kepercayaan model jika Anda memiliki kontrolnya.")

else:
    st.write("Silakan unggah gambar untuk memulai deteksi.")

st.markdown("---")
st.markdown("Aplikasi ini dibuat menggunakan [Streamlit](https://streamlit.io/) dan [Ultralytics YOLOv8](https://ultralytics.com/).")