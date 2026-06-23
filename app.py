import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import requests
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import io

# ── Config ────────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.75
CLASS_NAME     = "Transformer"
PAGE_TITLE     = "Transformer Detection — NBPower"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🔌",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1f77b4; }
    .metric-label { font-size: 0.85rem; color: #666; }
    .stProgress > div > div { background-color: #1f77b4; }
</style>
""", unsafe_allow_html=True)

# ── Load model (cache par chemin) ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, image: np.ndarray, conf: float):
    results = model.predict(
        source=image,
        conf=conf,
        device="mps",   # → "cpu" sur Streamlit Cloud
        verbose=False,
    )
    return results[0]

def draw_boxes(image: np.ndarray, result) -> np.ndarray:
    img = image.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 120, 255), 2)
        label = f"{CLASS_NAME} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 120, 255), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img

def build_dataframe(result) -> pd.DataFrame:
    rows = []
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        rows.append({
            "ID": i + 1,
            "Confiance": f"{conf:.2%}",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "Largeur (px)": x2 - x1,
            "Hauteur (px)": y2 - y1,
        })
    return pd.DataFrame(rows)

def show_results(image_bgr: np.ndarray, result):
    n     = len(result.boxes)
    confs = [float(b.conf[0]) for b in result.boxes]
    avg   = np.mean(confs) if confs else 0.0
    best  = max(confs)     if confs else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{n}</div>
            <div class="metric-label">Transformateurs détectés</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{avg:.0%}</div>
            <div class="metric-label">Confiance moyenne</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{best:.0%}</div>
            <div class="metric-label">Meilleure confiance</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    annotated_rgb = cv2.cvtColor(draw_boxes(image_bgr, result), cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="Résultat de détection", use_container_width=True)

    if n > 0:
        st.markdown("#### Détails par détection")
        df = build_dataframe(result)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter CSV", csv, "detections.csv", "text/csv")
    else:
        st.info("Aucun transformateur détecté avec ce seuil de confiance.")

# ── UI principale ─────────────────────────────────────────────────────────────
st.title("🔌 Transformer Detection")
st.caption("Détection automatique de transformateurs sur poteau — NBPower")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paramètres")

    # ── Chargement du modèle ──────────────────────────────────────────────────
    st.markdown("### 🧠 Modèle")

    model_source = st.radio(
        "Source",
        ["Chemin local", "Upload fichier"],
        horizontal=True,
    )

    model_path = None

    if model_source == "Chemin local":
        model_path_input = st.text_input(
            "Chemin vers le fichier .pt",
            value="runs/transformer-v1/weights/best.pt",
            placeholder="runs/mon-modele/weights/best.pt",
        )
        if model_path_input and Path(model_path_input).exists():
            model_path = model_path_input
        elif model_path_input:
            st.error("Fichier introuvable.")

    else:  # Upload fichier
        uploaded_model = st.file_uploader(
            "Glisser-déposer un fichier .pt",
            type=["pt"],
        )
        if uploaded_model:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            tmp.write(uploaded_model.read())
            tmp.flush()
            model_path = tmp.name
            st.success(f"✅ {uploaded_model.name}")

    st.markdown("---")

    # ── Seuil de confiance ────────────────────────────────────────────────────
    conf = st.slider(
        "Seuil de confiance",
        min_value=0.1, max_value=1.0,
        value=CONF_THRESHOLD, step=0.05,
        help="0.75 = optimal (F1 max)"
    )
    st.markdown("**Seuils recommandés**")
    st.markdown("- `0.75` — meilleur F1 (production)")
    st.markdown("- `0.856` — zéro faux positif (alertes)")

    # ── Info modèle actif ─────────────────────────────────────────────────────
    if model_path:
        st.markdown("---")
        st.markdown("**Modèle actif**")
        name = uploaded_model.name if model_source == "Upload fichier" and uploaded_model else Path(model_path).name
        st.code(name)

# ── Chargement du modèle ──────────────────────────────────────────────────────
if not model_path:
    st.warning("⬅️ Charge un modèle dans la sidebar pour commencer.")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

# ── Onglets ───────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_url = st.tabs(["📷 Image", "🎬 Vidéo", "🌐 URL"])

# ── Tab Image ─────────────────────────────────────────────────────────────────
with tab_img:
    uploaded = st.file_uploader(
        "Glisser-déposer une image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        with st.spinner("Analyse en cours..."):
            result = run_inference(model, image_bgr, conf)
        show_results(image_bgr, result)

# ── Tab Vidéo ─────────────────────────────────────────────────────────────────
with tab_vid:
    uploaded_vid = st.file_uploader(
        "Glisser-déposer une vidéo",
        type=["mp4", "mov", "avi", "mkv"],
    )
    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile.flush()

        cap          = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25

        st.info(f"Vidéo : {total_frames} frames à {fps:.0f} fps")
        process_every = st.slider("Analyser 1 frame sur N", 1, 30, 5)

        if st.button("▶️ Lancer l'analyse"):
            frame_placeholder = st.empty()
            progress          = st.progress(0)
            all_detections    = []
            frame_idx         = 0
            processed         = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % process_every == 0:
                    result        = run_inference(model, frame, conf)
                    annotated     = draw_boxes(frame, result)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_rgb, use_container_width=True)
                    for box in result.boxes:
                        all_detections.append({
                            "Frame": frame_idx,
                            "Temps (s)": round(frame_idx / fps, 2),
                            "Confiance": f"{float(box.conf[0]):.2%}",
                        })
                    processed += 1
                progress.progress(min(frame_idx / max(total_frames, 1), 1.0))
                frame_idx += 1

            cap.release()
            progress.progress(1.0)
            st.success(f"✅ Analyse terminée — {processed} frames traitées")

            if all_detections:
                df_vid = pd.DataFrame(all_detections)
                st.dataframe(df_vid, use_container_width=True)
                csv = df_vid.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Exporter CSV", csv, "detections_video.csv", "text/csv")
            else:
                st.info("Aucun transformateur détecté dans la vidéo.")

# ── Tab URL ───────────────────────────────────────────────────────────────────
with tab_url:
    url = st.text_input("URL de l'image", placeholder="https://example.com/image.jpg")
    if url and st.button("🔍 Analyser"):
        try:
            with st.spinner("Téléchargement..."):
                response  = requests.get(url, timeout=10)
                response.raise_for_status()
                image_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
                image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            with st.spinner("Analyse en cours..."):
                result = run_inference(model, image_bgr, conf)
            show_results(image_bgr, result)
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors du téléchargement : {e}")
        except Exception as e:
            st.error(f"Erreur : {e}")
