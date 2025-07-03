import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import base64
import os

# ========== CONFIGURASI HALAMAN ==========
st.set_page_config(
    page_title="YOLOv8 - Deteksi Objek",
    layout="wide",
    page_icon="🧠"
)

# ========== TEMA TERANG / GELAP ==========
dark_mode = st.sidebar.toggle("🌙 Mode Gelap")
background_color = "#1e1e1e" if dark_mode else "#f9f9f9"
text_color = "#ffffff" if dark_mode else "#000000"
card_color = "#2a2a2a" if dark_mode else "#ffffff"

# ========== STYLING MODERN ==========
st.markdown(f"""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
        background-color: {background_color};
        color: {text_color};
        transition: all 0.3s ease;
    }}
    .stApp {{
        background-color: {background_color};
    }}
    .block-container {{
        padding: 2rem 2.5rem;
    }}
    h1, h2, h3 {{
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: {text_color};
    }}
    p, li, label, span {{
        font-size: 1rem;
        line-height: 1.6;
        color: {text_color};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 16px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        color: #666;
        border: 1px solid #ccc;
        border-radius: 8px 8px 0 0;
        margin-right: 5px;
        background-color: #eaeaea;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #1f77b4;
        color: #fff;
        border-bottom: 3px solid #1f77b4;
    }}
    .stTabs [aria-selected="false"]:hover {{
        background-color: #dcdcdc;
        color: #000;
    }}
    .stButton > button {{
        background-color: #1f77b4;
        color: #fff;
        font-weight: 500;
        border-radius: 10px;
        padding: 0.6rem 1.3rem;
        border: none;
        transition: background 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: #145a86;
    }}
    img {{
        border-radius: 12px;
        margin-bottom: 1rem;
    }}
    .dataframe {{
        background-color: {card_color};
        color: {text_color};
        border-radius: 10px;
    }}
    .stDownloadButton button {{
        background-color: #28a745;
        color: white;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1.2rem;
    }}
    .stDownloadButton button:hover {{
        background-color: #218838;
    }}
    hr {{
        border: none;
        height: 1px;
        background: #ccc;
        margin: 2rem 0;
    }}
    a {{
        color: #1f77b4;
        font-weight: 500;
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    </style>
""", unsafe_allow_html=True)

# ========== JUDUL UTAMA ==========
st.markdown(f"""
<div style='text-align:center; padding: 1rem;'>
    <h1 style='margin-bottom: 0.2em;'>🧠 Deteksi Objek YOLOv8</h1>
    <p style='font-size:1.1rem; color:{text_color};'>
        Unggah gambar atau video untuk mendeteksi objek secara otomatis dan cepat
    </p>
</div>
""", unsafe_allow_html=True)

# ========== LOAD YOLO MODEL ==========
model = YOLO("yolov8n.pt")

# ========== TAB INPUT ==========
tab1, tab2 = st.tabs(["📷 Deteksi Gambar", "🎞️ Deteksi Video"])

# ========== TAB GAMBAR ==========
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_img = st.file_uploader("📷 Unggah Gambar (jpg/png)", type=["jpg", "jpeg", "png"])
        if uploaded_img:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="📸 Gambar Asli", use_column_width=True)

            img_array = np.array(image)
            results = model(img_array)[0]
            annotated_img = results.plot()

            labels = results.names
            detections = results.boxes.data.cpu().numpy()
            detection_df = pd.DataFrame(detections, columns=["x1", "y1", "x2", "y2", "confidence", "class"])
            detection_df["label"] = detection_df["class"].apply(lambda x: labels[int(x)])
            detection_df = detection_df[["label", "confidence", "x1", "y1", "x2", "y2"]]

            st.download_button("⬇️ Unduh Gambar Hasil", data=cv2.imencode('.jpg', annotated_img)[1].tobytes(),
                               file_name="hasil_deteksi.jpg", mime="image/jpeg")

    with col2:
        if uploaded_img:
            st.image(annotated_img, caption="🔍 Gambar dengan Deteksi", use_column_width=True)
            st.markdown("### 📋 Data Deteksi")
            filter_label = st.selectbox("Filter Label", options=["Semua"] + sorted(detection_df["label"].unique().tolist()))
            if filter_label != "Semua":
                st.dataframe(detection_df[detection_df["label"] == filter_label])
            else:
                st.dataframe(detection_df)

# ========== TAB VIDEO ==========
with tab2:
    uploaded_video = st.file_uploader("🎬 Unggah Video (mp4/avi/mov)", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        st.video(tfile.name)
        st.info("⚙️ Memproses video, mohon tunggu...")

        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = "output_detected.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        progress = st.progress(0)
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            annotated_frame = results.plot()
            out.write(annotated_frame)
            count += 1
            progress.progress(min(count / total_frames, 1.0))

        cap.release()
        out.release()

        st.success("✅ Deteksi selesai! Tonton hasilnya:")
        st.video(output_path)

        with open(output_path, "rb") as file:
            video_bytes = file.read()
            b64 = base64.b64encode(video_bytes).decode()
            href = f'<a href="data:video/mp4;base64,{b64}" download="hasil_deteksi.mp4">📥 Unduh Video Hasil</a>'
            st.markdown(href, unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown(f"""
<hr style="margin-top: 50px;">
<div style="text-align:center; color:{text_color}; font-size: 0.9rem;">
    Dibuat dengan 💡 faulnam | YOLOv8 Deployment | Streamlit UI
</div>
""", unsafe_allow_html=True)
