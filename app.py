import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "mobilenetv2_cats_dogs.keras"
)

IMG_SIZE = (160, 160)

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(ellipse at 10% 0%, #18334a 0%, transparent 42%), radial-gradient(ellipse at 90% 15%, #30204c 0%, transparent 38%), linear-gradient(180deg, #0b1020 0%, #0d1428 100%);
    color: #f8fafc;
}
.block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 4rem; }
h1, h2, h3 { color: #f8fafc !important; letter-spacing: -0.7px; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(21, 29, 48, 0.85); border: 1px solid rgba(183, 247, 212, 0.15); border-radius: 24px; backdrop-filter: blur(12px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); transition: all 0.3s ease; }
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: rgba(183, 247, 212, 0.3); box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4); }
[data-testid="stFileUploader"] { background: rgba(21, 29, 48, 0.9); border: 2px dashed #60718d; border-radius: 20px; padding: 2rem 1.5rem; transition: all 0.3s ease; }
[data-testid="stFileUploader"]:hover { border-color: #b7f7d4; background: rgba(31, 39, 58, 0.9); transform: translateY(-2px); }
.stButton > button { background: linear-gradient(135deg, #b7f7d4 0%, #8de8f1 100%); color: #0b1020; border: none; border-radius: 14px; padding: 0.85rem 2rem; font-weight: 700; font-size: 16px; box-shadow: 0 4px 15px rgba(183, 247, 212, 0.3); transition: all 0.3s ease; text-transform: uppercase; }
.stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(183, 247, 212, 0.5); filter: brightness(1.1); }
[data-testid="stImage"] { border-radius: 18px; overflow: hidden; transition: transform 0.3s ease; }
[data-testid="stImage"] img { border-radius: 18px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4); }
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #b7f7d4 0%, #8de8f1 50%, #b7f7d4 100%); background-size: 200% 100%; animation: shimmer 2s infinite; border-radius: 10px; height: 8px; }
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }
@keyframes float { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-10px); } }
@keyframes glow { 0%, 100% { box-shadow: 0 12px 35px rgba(183, 247, 212, 0.2); } 50% { box-shadow: 0 12px 45px rgba(183, 247, 212, 0.4); } }
.block-container { animation: fadeUp 0.6s ease-out; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

def predict_image(image, model):
    image = image.convert("RGB").resize(IMG_SIZE)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_batch = np.expand_dims(image_array, axis=0)
    dog_probability = float(model.predict(image_batch, verbose=0)[0][0])
    if not np.isfinite(dog_probability):
        raise ValueError("The model returned an invalid prediction.")
    dog_probability = float(np.clip(dog_probability, 0.0, 1.0))
    if dog_probability >= 0.5:
        return "Dog", dog_probability
    return "Cat", 1.0 - dog_probability

# Title section
st.markdown("### 🐾 PAWVISION / COMPUTER VISION")
st.title("Every picture has a pawprint")
st.markdown("*A deep-learning image classifier powered by MobileNetV2 that distinguishes cats from dogs with high accuracy and confidence.*")
st.markdown("---")

uploaded_file = st.file_uploader("📁 Choose an image to classify", type=["jpg", "jpeg", "png", "webp"], help="Upload a clear photo of a cat or dog for best results")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded image", use_container_width=True)

        if st.button("🔍 Classify Image", type="primary"):
            try:
                model = load_model()
                with st.spinner("🧠 Analyzing image..."):
                    predicted_class, confidence = predict_image(image, model)
                st.success("✅ Classification complete!")

                col1, col2 = st.columns([1.2, 1], gap="large")
                with col1:
                    st.markdown("#### 🖼️ Your Image")
                    st.image(image, use_container_width=True)

                with col2:
                    emoji = "🐶" if predicted_class == "Dog" else "🐱"
                    st.markdown("#### 🎯 Classification Result")
                    st.markdown(f'<div style="background: linear-gradient(145deg, rgba(31, 47, 68, 0.98), rgba(21, 29, 48, 0.98)); border: 1px solid rgba(183, 247, 212, 0.35); border-radius: 24px; padding: 32px 28px; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25); margin-bottom: 20px;"><div style="font-size: 64px; text-align: center; margin-bottom: 12px;">{emoji}</div><div style="color: #f8fafc; font-size: 36px; font-weight: 800; text-align: center; margin: 12px 0;">{predicted_class}</div><div style="color: #94a3b8; font-size: 12px; text-align: center; font-weight: 600; text-transform: uppercase; margin-bottom: 24px;">Predicted Class</div><div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(183, 247, 212, 0.3), transparent); margin: 24px 0;"></div><div style="color: #b7f7d4; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px;">Model Confidence</div><div style="color: #f8fafc; font-size: 32px; font-weight: 800;">{confidence:.1%}</div></div>', unsafe_allow_html=True)
                    st.progress(confidence)
                    confidence_level = "High" if confidence > 0.9 else "Good" if confidence > 0.7 else "Moderate"
                    st.caption(f"🎓 **Confidence Level:** {confidence_level}  \nℹ️ The model is {confidence:.1%} certain about this prediction.")
            except FileNotFoundError as e:
                st.error(f"❌ **Model Error:** {e}")
            except (OSError, ValueError) as e:
                st.error(f"❌ **Classification Error:** {e}")
            except Exception as e:
                st.error("❌ **Unexpected Error:** An error occurred during classification.")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("❌ **Invalid Image:** Please upload a valid JPG, PNG, or WebP file.")
