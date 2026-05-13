#  SKIN LESION SEVERITY CLASSIFIER
** used : [Python], [TensorFlow], [Streamlit], [Keras] **

***Developed By:*** JAKIA SULTANA SUMA

----------------------------------------------------------------------------------

# Project Overview: 

Developed a highly interactive Streamlit web application for automated skin lesion analysis using a CNN model trained on the HAM10000 dataset. The system classifies Benign and Malignant skin lesions through real-time image upload and instant prediction, supporting early-stage skin cancer screening and medical triaging.

The HAM10000 dataset consists of 7 diagnostic categories, which this project re-classifies into a binary system (Benign vs. Malignant) for the severity classifier.

The 7 original image categories are:

 - **Melanoma (MEL):** A highly malignant and dangerous form of skin cancer.
 - **Melanocytic Nevi (NV):** Common benign moles; the most frequent category in the dataset.
 - **Basal Cell Carcinoma (BCC):** A common type of skin cancer that is malignant but rarely spreads.
 - **Actinic Keratoses (AKIEC):** Pre-cancerous or early-stage malignant lesions (Bowen's disease).
 - **Benign Keratosis-like Lesions (BKL):** Non-cancerous lesions including seborrheic keratoses and solar lentigines.
 - **Dermatofibroma (DF):** A common benign fibrous growth on the skin.
 - **Vascular Lesions (VASC):** Benign lesions such as cherry angiomas or angiokeratomas.

----------------------------------------------------------------------------------

## Key Features: 

 - **Advanced CNN Architecture:** Utilizes deep convolutional layers, Batch Normalization, and progressive Dropout to prevent overfitting, ensuring the model generalizes well to new, unseen skin lesion images.

 - **Data Augmentation:** Implements random flips, rotations, zooming, and translations during the training phase to boost model robustness and help manage the class imbalance between benign and malignant samples.

- **Highly Interactive Streamlit Web Dashboard:** A clean, user-friendly frontend that allows users to upload dermoscopy images and instantly view the model's prediction (Benign vs. Malignant) alongside a confidence probability distribution.

- **Training Analytics:** The dashboard is designed to plot interactive line charts of the model's training accuracy and loss over time, providing transparency into how the model learned to classify lesion severity.





