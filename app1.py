import streamlit as st
import numpy as np
from PIL import Image
import os

# ---------------- TENSORFLOW SAFE IMPORT ----------------
try:
    import tensorflow as tf
    HAS_TF = True
except Exception as e:
    HAS_TF = False
    TF_ERROR = str(e)
    tf = None

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="Skin Lesion Severity Classifier",
    layout="centered"
)

# ---------------- CLASS LABELS ----------------
CLASS_NAMES = [
    'Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
    'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions'
]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_file():
    if not HAS_TF:
        return None

    # Define the model filename
    model_name = "skin_lesion.h5"
    
    # Absolute path to the directory where app1.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, model_name)

    # Fallback check for current working directory
    if not os.path.exists(model_path):
        model_path = model_name 

    if not os.path.exists(model_path):
        return "FILE_NOT_FOUND"

    try:
        model = tf.keras.models.load_model(model_path)
        return model, os.path.basename(model_path)
    except Exception as e:
        return f"ERROR: {str(e)}"

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- MAIN APP ----------------
def main():
    # Sidebar
    with st.sidebar:
        st.title("About Project")
        st.info("Skin Lesion Severity Classifier (~79% Accuracy)")
        st.write("This model classifies skin lesions into 7 categories based on the HAM10000 dataset.")
        st.markdown("---")
        st.markdown("**Developer:** Jakia Sultana Suma")

    st.title("Skin Lesion Severity Classifier")
    st.write("Upload a clear image of a skin lesion, and the model will predict its category.")

    # Check TensorFlow
    if not HAS_TF:
        st.error(f"TensorFlow Load Error: {TF_ERROR}")
        st.stop()

    # Load Model
    model_result = load_model_file()

    if model_result == "FILE_NOT_FOUND":
        st.error("Error: Model file 'skin_lesion.h5' not found in the directory.")
        st.info("Please make sure your model file is in the same folder as this script.")
        # Debugging info
        st.write(f"Current Directory: {os.getcwd()}")
        st.stop()
    elif isinstance(model_result, str) and model_result.startswith("ERROR"):
        st.error(f"Model Loading Failed: {model_result}")
        st.stop()
    else:
        model, model_name = model_result
        st.caption(f"Loaded model: {model_name}")

    # Upload Image
    uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                processed = preprocess_image(image)
                predictions = model.predict(processed)[0]
                
                class_index = np.argmax(predictions)
                confidence = predictions[class_index]

                st.success(f"**Result: {CLASS_NAMES[class_index]}**")
                st.info(f"**Confidence Level: {confidence * 100:.2f}%**")

                st.write("### Probabilities for All Classes")
                for name, prob in zip(CLASS_NAMES, predictions):
                    col1, col2 = st.columns([2, 3])
                    col1.text(name)
                    col2.progress(float(prob))

if __name__ == "__main__":
    main()