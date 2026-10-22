"""ChestScope demo.  Run from the repo root:  streamlit run app.py"""
import streamlit as st
from PIL import Image

from src.predict import load_predictor, predict_pil


st.set_page_config(page_title="ChestScope", page_icon="🫁", layout="wide")

st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0b0d 55%, #0d0d0d 100%);
    color: #f5f5f5;
}

/* Main title */
h1 {
    color: #ff3030 !important;
    font-weight: 800 !important;
    text-shadow: 0 0 18px rgba(255, 48, 48, 0.25);
}

/* Section headings */
h2, h3 {
    color: #ff5555 !important;
}

/* Warning banner */
.stAlert {
    background-color: #240d10;
    border: 1px solid #b91c2c;
    border-radius: 10px;
    color: #f5f5f5;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #151515;
    border: 1px dashed #ff3030;
    border-radius: 14px;
    padding: 18px;
}

/* Buttons */
.stButton > button {
    background: #d71920;
    color: white;
    border: 1px solid #ff4545;
    border-radius: 8px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #ff3030;
    color: white;
    border-color: #ff7777;
    box-shadow: 0 0 14px rgba(255, 48, 48, 0.4);
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1c1c1c, #270d10);
    border: 1px solid #682027;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #d0d0d0 !important;
}

[data-testid="stMetricValue"] {
    color: #ff4545 !important;
    font-weight: 800;
}

/* Prediction and status text */
[data-testid="stAlert"] p {
    color: #f5f5f5;
}

/* Uploaded images */
[data-testid="stImage"] img {
    border: 1px solid #6e2028;
    border-radius: 10px;
}

/* Informational text */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #b8b8b8 !important;
}

/* Dividers */
hr {
    border-color: #522027;
}

/* Sidebar, if used */
[data-testid="stSidebar"] {
    background-color: #101010;
    border-right: 1px solid #5b171f;
}

/* Input focus */
[data-testid="stFileUploader"]:focus-within {
    box-shadow: 0 0 12px rgba(255, 48, 48, 0.25);
}

/* Smooth page entrance */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes redGlow {
    0%, 100% {
        box-shadow: 0 0 6px rgba(255, 48, 48, 0.15);
    }
    50% {
        box-shadow: 0 0 20px rgba(255, 48, 48, 0.4);
    }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.55; }
}

h1,
[data-testid="stImage"],
[data-testid="stMetric"],
[data-testid="stFileUploader"] {
    animation: fadeIn 0.65s ease-out both;
}

[data-testid="stMetric"] {
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    animation: redGlow 2s infinite ease-in-out;
}

[data-testid="stImage"] img {
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

[data-testid="stImage"] img:hover {
    transform: scale(1.015);
    box-shadow: 0 0 16px rgba(255, 48, 48, 0.25);
}

/* Animated processing indicator */
[data-testid="stSpinner"] {
    color: #ff4545 !important;
}

[data-testid="stSpinner"] svg {
    stroke: #ff3030 !important;
}

/* Respect reduced-motion accessibility settings */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("ChestScope: Chest X-Ray Pneumonia Triage Support")
st.warning("**Research prototype, not a diagnostic device.** Outputs are not medical advice and must not be used "
           "for clinical decisions. Trained on a small public pediatric dataset from a single institution.")


@st.cache_resource
def get_model():
    return load_predictor("cpu")


try:
    model, cam_obj, threshold = get_model()
except Exception as e:
    st.error(f"Could not load the model (expected results/checkpoints/final.pt). Details: {e}")
    st.stop()

file = st.file_uploader("Upload a chest X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])
if file is not None:
    with st.spinner("🔴 Upload received — analyzing X-ray..."):
        img = Image.open(file)
        out = predict_pil(model, cam_obj, threshold, img)

    st.toast("X-ray analysis complete!", icon="✅")

    c1, c2 = st.columns(2)
    c1.image(out["gray"], caption="Input (resized to 224x224)", use_container_width=True)
    c2.image(out["overlay"], caption="Grad-CAM: regions driving the pneumonia score", use_container_width=True)

    (st.error if out["label"].startswith("Pneumonia") else st.success)(f"**{out['label']}**")
    m1, m2 = st.columns(2)
    m1.metric("P(pneumonia)", f"{out['prob']*100:.1f}%")
    m2.metric("Confidence in this call", f"{out['confidence']*100:.1f}%")
    st.caption(f"Decision threshold {threshold:.3f}, chosen on a validation set to reach about 95% recall. "
               "Heatmaps show where the model looked, which is not proof of a finding.")

else:
    st.info("Upload an image to see the prediction, confidence score, and heatmap.")