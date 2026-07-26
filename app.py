import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["Acne", "Rosacea"]
MODEL_PATH = "models/acne_rosacea_mobilenetv3.keras"

st.set_page_config(page_title="Acne vs Rosacea Classifier", page_icon="🔬", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict(model, pil_image):
    """Return the predicted label and the probability for each class."""
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    probs = model.predict(arr, verbose=0)[0]          # e.g. [P(Acne), P(Rosacea)]
    predicted_idx = int(np.argmax(probs))
    return CLASS_NAMES[predicted_idx], probs

st.title("Acne vs Rosacea Classifier")
st.caption(
    "A coursework CNN model (GET 324) trained to distinguish Acne from Rosacea in skin photos. "
    "Educational project only — not a medical diagnostic tool."
)

model = load_model()
uploaded_file = st.file_uploader("Upload a skin photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300)

    label, probs = predict(model, img)
    st.write(f"**Prediction: {label}**")
    for name, p in zip(CLASS_NAMES, probs):
        st.progress(int(p * 100), text=f"{name}: {p * 100:.1f}%")
