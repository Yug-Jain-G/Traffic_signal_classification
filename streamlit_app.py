import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px

# ======================
# Load trained model
# ======================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("traffic_sign_model.keras")

model = load_model()

CLASS_NAMES = [
    'Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)',
    'Speed limit (70km/h)','Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)',
    'Speed limit (120km/h)','No passing','No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection','Priority road','Yield','Stop','No vehicles',
    'Vehicles over 3.5 metric tons prohibited','No entry','General caution','Dangerous curve to the left',
    'Dangerous curve to the right','Double curve','Bumpy road','Slippery road','Road narrows on the right',
    'Road work','Traffic signals','Pedestrians','Children crossing','Bicycles crossing','Beware of ice/snow',
    'Wild animals crossing','End of all speed and passing limits','Turn right ahead','Turn left ahead',
    'Ahead only','Go straight or right','Go straight or left','Keep right','Keep left','Roundabout mandatory',
    'End of no passing','End of no passing by vehicles over 3.5 metric tons'
]

# ======================
# Page Config & CSS
# ======================
st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦", layout="wide")

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #2b5876, #4e4376);
        color: white;
    }
    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #f8f9fa;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2em;
        color: #d6d6d6;
        margin-top: 0;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# Header
# ======================
st.markdown("<div class='main-title'>🚦 Traffic Sign Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload a traffic sign image and let AI predict its meaning</div>", unsafe_allow_html=True)
st.markdown("---")

# ======================
# Sidebar
# ======================
st.sidebar.header("ℹ️ About")
st.sidebar.write(
    """
    This app uses a **Convolutional Neural Network (CNN)** trained on  
    the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.  

    Upload a traffic sign image, and the model will predict which sign it is.
    """
)
st.sidebar.info("Tip: Try uploading official traffic sign images for best results.")

# ======================
# File uploader
# ======================
uploaded_file = st.file_uploader("📤 Upload a traffic sign image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    # Show uploaded image (smaller + centered)
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.image(image, caption="🖼 Uploaded Image", width=250)
        st.markdown("</div>", unsafe_allow_html=True)

    # Preprocess
    img = image.resize((50, 50))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display Prediction Card (centered)
    with col2:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center; align-items:center; height:100%;">
                <div class='prediction-card' style="width:100%; max-width:400px; margin:auto;">
                    <h2 style="color:#00FF7F; margin-bottom:10px;">✅ Prediction</h2>
                    <h1 style="color:#FFD700; font-size:2.5em; margin:0;">{predicted_class}</h1>
                    <p style="font-size:1.1em;"><b>Confidence:</b> {confidence:.2%}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Confidence probabilities (interactive bar chart)
    with st.expander("🔍 View Prediction Probabilities"):
        probs = predictions[0]
        fig = px.bar(
            x=list(range(len(CLASS_NAMES))),
            y=probs,
            labels={'x': "Class Index", 'y': "Probability"},
            title="Class Probabilities"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Upload an image to get started!")
