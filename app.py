import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os

# --- Konfigurasi Model ---
MODEL_PATH = "best.pt"

# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model dari {path}. Pastikan file 'best.pt' ada di folder yang sama. Error: {e}")
        st.stop()

model = load_yolo_model(MODEL_PATH)

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="KopiLens: Deteksi Biji Kopi Premium",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- CSS Kustom untuk Tema Warna & Background Coklat Variatif ---
st.markdown("""
    <style>
    /* Background utama seluruh aplikasi */
    html, body, .stApp {
        background-color: #4E342E !important; /* Coklat tua */
        color: #FBE9E7 !important; /* Teks terang */
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #6D4C41 !important; /* Coklat medium */
        color: #FBE9E7 !important;
        border-right: 3px solid #8D6E63;
    }

    /* Kontainer utama */
    .block-container {
        background-color: rgba(141, 110, 99, 0.2) !important; /* Coklat muda semi transparan */
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(62, 39, 35, 0.4);
    }

    /* Judul utama */
    h1 {
        color: #FFCC80 !important;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 3.2rem;
        margin-bottom: 0.5rem;
    }

    /* Subjudul */
    h3 {
        color: #FFE0B2 !important;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.6rem;
        margin-top: 0;
    }

    /* Paragraf */
    p, .stMarkdown {
        color: #FBE9E7 !important;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    /* Tombol upload file */
    button[kind="file"] {
        background-color: #FFB74D !important; /* Coklat terang / oranye */
        color: #3E2723 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        font-size: 1rem !important;
        font-weight: bold !important;
    }

    button[kind="file"]:hover {
        background-color: #FFA726 !important;
        color: #3E2723 !important;
    }

    /* Informasi (info box Streamlit) */
    .stAlert {
        background-color: #D7CCC8 !important;
        border-left: 6px solid #8D6E63 !important;
        color: #3E2723 !important;
    }

    /* Garis pemisah */
    hr {
        border-top: 2px solid #8D6E63;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* Caption gambar */
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Footer */
    footer {
        background-color: #3E2723 !important;
        padding: 16px;
        margin-top: 2rem;
        text-align: center;
        color: #FBE9E7 !important;
        border-top: 2px solid #6D4C41;
    }

    footer a {
        color: #FFCC80;
        text-decoration: none;
        font-weight: bold;
    }

    footer a:hover {
        text-decoration: underline;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #4E342E;
    }

    ::-webkit-scrollbar-thumb {
        background: #8D6E63;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)


# --- Judul dan Deskripsi Aplikasi ---
title_placeholder = st.empty()
title_placeholder.markdown("<h1 style='text-align: center;'>☕ KopiLens</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Deteksi & Klasifikasi Kualitas Biji Kopi Premium</h3>", unsafe_allow_html=True)
st.write("---")
st.markdown(
    """
    Selamat datang di **KopiLens**, alat cerdas Anda untuk menganalisis biji kopi!
    Unggah gambar biji kopi Anda dan biarkan teknologi AI kami yang canggih
    dengan cepat mengidentifikasi dan mengklasifikasikan berbagai kondisi biji.
    Dapatkan wawasan akurat untuk kualitas kopi yang lebih baik.
    """
)
st.write("")


# --- Komponen Upload Gambar ---
st.subheader("Mulai Analisis Biji Kopi Anda:")
uploaded_file = st.file_uploader("Pilih gambar biji kopi untuk dianalisis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Menganalisis gambar Anda, mohon bersabar...☕'):
        image = Image.open(uploaded_file)

        col_original, col_detected = st.columns(2)

        with col_original:
            st.markdown("#### Gambar Asli")
            st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        results = model.predict(source=image_bgr, save=False, conf=0.25, iou=0.7, verbose=False)

        predictions_found = False
        detected_objects_info = []

        for r in results:
            annotated_image_bgr = r.plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

            with col_detected:
                st.markdown("#### Hasil Deteksi KopiLens")
                st.image(annotated_image_rgb, caption="Deteksi Model AI", use_column_width=True)

            if len(r.boxes) > 0:
                predictions_found = True
                st.markdown("---")
                st.markdown("#### Temuan Analisis:")
                for box in r.boxes:
                    class_id = int(box.cls)
                    conf = float(box.conf)
                    class_name = model.names.get(class_id, f"Class {class_id}")
                    st.markdown(f"- **{class_name.replace('_', ' ').title()}** (Kepercayaan: `{conf:.2f}`)")

        if not predictions_found:
            st.warning("Maaf, KopiLens tidak dapat mendeteksi biji kopi apa pun dalam gambar ini. 😔")
            st.info("💡 Pastikan gambar biji kopi jelas, tidak terlalu gelap, dan biji terlihat dengan baik. Coba gambar lain!")

else:
    st.info("☝️ Unggah gambar biji kopi Anda di sini (JPG, JPEG, PNG) untuk memulai analisis!")


# --- FOOTER KUSTOM DI BAGIAN BAWAH HALAMAN UTAMA ---
st.markdown("---")  # Garis pemisah sebelum footer
st.markdown(
    """
    <style>
        .footer {
            background-color: #6D4C41;
            padding: 20px;
            margin-top: 3rem;
            border-radius: 10px;
            color: #FBE9E7;
            width: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }
    </style>

    <div class="footer">
        <p>Menggunakan Streamlit & Ultralytics YOLOv8</p>
        <p style='font-size: small;'>Untuk tujuan edukasi dan demonstrasi.</p>
    </div>
    """,
    unsafe_allow_html=True
)

