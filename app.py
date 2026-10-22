"""ChestScope demo.  Run from the repo root:  streamlit run app.py"""
import streamlit as st
from PIL import Image

from src.predict import load_predictor, predict_pil

st.set_page_config(page_title="ChestScope", page_icon="🫁", layout="wide")
st.title("ChestScope: chest X-ray pneumonia triage support")
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
    img = Image.open(file)
    out = predict_pil(model, cam_obj, threshold, img)
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