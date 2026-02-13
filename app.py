import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Alzheimer Detection System", page_icon="🧠", layout="centered")
# 🎨 Medical Theme Styling
st.markdown("""
<style>
body {
    background-color: #f4f8fb;
}

.main {
    background-color: #f4f8fb;
}

h1 {
    color: #0b3c5d;
}

h2, h3 {
    color: #1d6fa5;
}

.stButton>button {
    background-color: #1d6fa5;
    color: white;
    border-radius: 8px;
}

.stProgress > div > div > div > div {
    background-color: #1d6fa5;
}

.stAlert {
    border-radius: 10px;
}

.stFileUploader {
    border-radius: 10px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# 🔹 Arabic Main Title فقط
st.title("🧠 نظام تشخيص مرض الزهايمر الذكي")

st.markdown("""
This application uses a fine-tuned **ResNet50** model to analyze MRI brain scans  
and classify the condition into one of **four Alzheimer's stages**.
""")

st.info("Current model validation accuracy: **≈ 72%**")

# 2. Load Model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('alzheimer_resnet50_final2.keras')

try:
    model = load_my_model()

    # IMPORTANT: Must match training order
    class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

    # English labels
    class_labels = {
        'Non_Demented': 'Non Demented (Healthy)',
        'Very_Mild_Demented': 'Very Mild Dementia',
        'Mild_Demented': 'Mild Dementia',
        'Moderate_Demented': 'Moderate Dementia',
    }

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Upload MRI Image
uploaded_file = st.file_uploader("Upload MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    img = Image.open(uploaded_file).convert('RGB')

    with col1:
        st.image(img, caption='Uploaded Image', use_container_width=True)

    # 4. Prediction
    with st.spinner('Analyzing MRI scan...'):
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)[0]
        idx = int(np.argmax(preds))
        prob = float(preds[idx])

        pred_class = class_names[idx]
        result_text = class_labels.get(pred_class, pred_class)

    # 5. Results
    with col2:
        st.subheader("Diagnosis Result:")

        if pred_class == 'Non_Demented':
            st.success(result_text)
        else:
            st.warning(result_text)

        st.write(f"Confidence: {prob * 100:.2f}%")
        st.progress(prob)

        st.markdown("### All Class Probabilities:")
        sorted_pairs = sorted(zip(class_names, preds), key=lambda x: x[1], reverse=True)
        for cls, p in sorted_pairs:
            label = class_labels.get(cls, cls)
            st.write(f"- **{label}**: {p*100:.2f}%")

st.markdown("---")
st.caption("Disclaimer: This model is for educational and research purposes only and should not replace professional medical diagnosis.")
