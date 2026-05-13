#  SKIN LESION SEVERITY CLASSIFIER
** used : [Python], [TensorFlow], [Streamlit], [Keras] **

***Developed By:*** JAKIA SULTANA SUMA

----------------------------------------------------------------------------------

# Project Overview: 

Developed a highly interactive Streamlit web application for automated skin lesion analysis using a CNN model trained on the HAM10000 dataset. The system classifies Benign and Malignant skin lesions through real-time image upload and instant prediction, supporting early-stage skin cancer screening and medical triaging.

The HAM10000 dataset consists of 7 diagnostic categories, which this project re-classifies into a binary system (Benign vs. Malignant) for the severity classifier.

*The 7 original image categories are:*

    - **Melanoma (MEL):** A highly malignant and dangerous form of skin cancer.
    - **Melanocytic Nevi (NV):** Common benign moles; the most frequent category in the dataset.
    - **Basal Cell Carcinoma (BCC):** A common type of skin cancer that is malignant but rarely spreads.
    - **Actinic Keratoses (AKIEC):** Pre-cancerous or early-stage malignant lesions (Bowen's disease).
    - **Benign Keratosis-like Lesions (BKL):** Non-cancerous lesions including seborrheic keratoses and solar lentigines.
    - **Dermatofibroma (DF):** A common benign fibrous growth on the skin.
    - **Vascular Lesions (VASC):** Benign lesions such as cherry angiomas or angiokeratomas.

----

## Key Features: 

    - **Advanced CNN Architecture:** Utilizes deep convolutional layers, Batch Normalization, and progressive Dropout to prevent overfitting, ensuring the model generalizes well to new, unseen skin lesion images.

    - **Data Augmentation:** Implements random flips, rotations, zooming, and translations during the training phase to boost model robustness and help manage the class imbalance between benign and malignant samples.

    - **Highly Interactive Streamlit Web Dashboard:** A clean, user-friendly frontend that allows users to upload dermoscopy images and instantly view the model's prediction (Benign vs. Malignant) alongside a confidence probability distribution.

    - **Training Analytics:** The dashboard is designed to plot interactive line charts of the model's training accuracy and loss over time, providing transparency into how the model learned to classify lesion severity.

----

## Project Structure

    - `app1.py:` The main Python script for the Streamlit web application, responsible for the dashboard interface and real-time classification.
    - `skin_lesion_severity.ipynb:` The original Jupyter Notebook containing the data exploration, preprocessing, and model training code.
    - `skin_lesion.h5:` The final trained Deep Learning model file (weights and architecture) used by the app for predictions.
    - `skin_lesion.pkl:` The serialized training history file, used to generate performance graphs on the dashboard.
    - `requirements.txt:` A list of all Python libraries (TensorFlow, OpenCV, etc.) needed to run the environment.
    - `run.bat`: A convenient Windows batch script to launch the Streamlit app locally with a single click.

---

##  How to Run 

### 1. Clone the repository
```bash
git clone https://github.com/jakia-suma/SKIN-LESION-SEVERITY-CLASSIFIER
cd skin_lesion
```
### 4. Run the Streamlit App
For a quick start on Windows, simply double-click the run.bat script. If you prefer using the command line, enter the following:
```bash
streamlit run app.py
```
Automatic browser redirection to the local host interface upon startup `http://localhost:8501`.

---

##  Model Architecture Summary

The Skin Lesion Severity Classifier uses an optimized CNN (build_optimized_model) designed for high-resolution medical imaging:
1. **Input:** 128 x 128 x 3  RGB images (Higher resolution to capture lesion textures). 
2. **Feature Extraction:**
    - 3 Convolutional Blocks: (64 - 128 - 256 filters).
    - Regularization: Includes L2 Regularization, Batch Normalization, and Dropout to prevent overfitting.
    - Downsampling: MaxPooling layers to extract dominant features.
3. **Classification Head:**
    - Global Average Pooling (GAP): Reduces parameters to maintain model efficiency.
    - Dense Layer: $512$ neurons for complex pattern recognition.
    - Output: Softmax layer for severity classification (Benign vs. Malignant).

4. **Optimizer:** Adam with a Cosine Decay learning rate schedule for precise convergence. 

---
*Developed as a technical showcase in Deep Learning and Computer Vision, focusing on the practical application of medical image classification.*




