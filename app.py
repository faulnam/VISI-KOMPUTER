import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import pandas as pd

# ====== Konfigurasi Halaman ======
st.set_page_config(page_title="Deteksi Objek Custom", layout="centered")

# ====== Tema Gelap/Terang ======
with st.sidebar:
    st.title("⚙️ Pengaturan")
    mode = st.radio("Pilih Mode Tema:", ("Terang", "Gelap"))
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Dibangun dengan YOLOv8 + Streamlit")

if mode == "Gelap":
    st.markdown("""
        <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# ====== Header ======
st.title("🧠 Deteksi Objek Custom dengan YOLOv8")
st.markdown("Silakan pilih mode input: **Gambar** atau **Video** dan jalankan deteksi objek menggunakan model Anda!")

# Load YOLO model
model = YOLO("yolov8n.pt")  # Pastikan model ada di folder

# Tabs: Gambar & Video
tab1, tab2 = st.tabs(["🖼️ Deteksi Gambar", "🎞️ Deteksi Video"])

# ========== TAB GAMBAR ===========
with tab1:
    st.subheader("📷 Upload Gambar")
    uploaded_img = st.file_uploader("Unggah gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        if uploaded_img.size > 5 * 1024 * 1024:
            st.warning("Ukuran gambar terlalu besar. Maksimum 5MB.")
            st.stop()

        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Gambar Asli", use_column_width=True)

        img_array = np.array(image)
        results = model(img_array)
        result_img = results[0].plot()

        st.subheader("🔍 Hasil Deteksi")
        st.image(result_img, caption="Gambar dengan Deteksi", use_column_width=True)

        # Detail Deteksi
        detection_data = results[0].boxes.data.cpu().numpy()
        if detection_data.size > 0:
            df = pd.DataFrame(detection_data, columns=["x1", "y1", "x2", "y2", "confidence", "class"])
            df["label"] = df["class"].apply(lambda x: model.names[int(x)])
            selected_labels = st.multiselect("Filter Label:", options=df["label"].unique())

            if selected_labels:
                df = df[df["label"].isin(selected_labels)]

            st.dataframe(df[["label", "confidence"]].sort_values("confidence", ascending=False))

        # Download hasil
        result_pil = Image.fromarray(result_img)
        tmp_download = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        result_pil.save(tmp_download.name)
        with open(tmp_download.name, "rb") as file:
            st.download_button("⬇️ Unduh Hasil Deteksi", data=file, file_name="hasil_deteksi.png", mime="image/png")

# ========== TAB VIDEO ===========
with tab2:
    st.subheader("🎬 Upload Video")
    uploaded_video = st.file_uploader("Unggah video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(uploaded_video)
        st.info("⚙️ Memproses video, mohon tunggu...")

        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = os.path.join(tempfile.gettempdir(), "output_detected.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            frame_num += 1
            progress.progress(min(frame_num / frame_count, 1.0))

        cap.release()
        out.release()

        st.success("✅ Deteksi selesai! Lihat hasilnya di bawah ini:")
        st.video(output_path)

        # Tombol unduh video
        with open(output_path, "rb") as file:
            st.download_button("⬇️ Unduh Video Hasil Deteksi", data=file, file_name="hasil_deteksi_video.mp4", mime="video/mp4")
