# AI-Powered Ovarian Ultrasound Diagnostic System

**Automated Detection of PCOS and Ovarian Cysts using Deep Learning**

## 📌 Project Overview
This project implements a Deep Learning solution to assist medical professionals in the automated diagnosis of ovarian conditions. Using a **Convolutional Neural Network (CNN)**, the system analyzes ultrasound images to classify them into 5 distinct categories, acting as a high-precision "second opinion" tool for medical triage.

The model distinguishes between pathological conditions (like PCOS and Complex Cysts) and normal biological states (Dominant Follicles), addressing a common challenge in manual ultrasound interpretation.

## 📖 Medical Terms Explained (What the AI Sees)

<img width="701" height="365" alt="image" src="https://github.com/user-attachments/assets/004f6945-09a1-4b12-a881-b5d5916d63ab" />

[Image of ultrasound ovarian cyst types]

### **1. Healthy Ovary**
* **What it is:** Normal ovarian tissue with no abnormalities.
* **Visuals:** Uniform gray texture. It may show small, transient follicles depending on the menstrual cycle phase, but lacks the specific patterns of cysts or PCOS.

### **2. Dominant Follicle**
* **What it is:** A *normal* physiological structure. Every month, one follicle grows larger than the others to release an egg (ovulation).
* **Visuals:** A single, clear, black circle (fluid-filled) on the ultrasound.
* **AI Challenge:** It looks very similar to a "Simple Cyst," making it the hardest class for AI (and humans) to distinguish.

### **3. Simple Cyst**
* **What it is:** A benign (harmless) fluid-filled sac that persists longer than a normal follicle.
* **Visuals:** A large, perfectly round, black circle with thin, smooth walls. Unlike a follicle, it does not disappear after ovulation.

### **4. Complex Cyst**
* **What it is:** A cyst that is not just clear fluid. It may contain blood, solid tissue, or septations (walls inside). These are potentially pathological (tumors, endometriosis, or dermoids) and require close medical monitoring.
* **Visuals:** A "messy" circle. Instead of being pure black, it has gray/white patches inside or irregular, thick borders.

### **5. PCOS (Polycystic Ovary Syndrome)**
* **What it is:** A hormonal disorder where the ovary develops many small, immature follicles that fail to release eggs.
* **Visuals:** The **"String of Pearls"** sign. You see multiple tiny black circles (follicles) arranged in a ring around the edge of the swollen ovary. This pattern is visually distinct from a single large cyst.

## 🚀 Key Features
* **High Accuracy:** Achieved **~99.5% accuracy** on the validation set.
* **Multi-Class Diagnosis:** Detects 5 specific classes: `PCOS`, `Complex Cyst`, `Simple Cyst`, `Dominant Follicle`, and `Healthy Ovary`.
* **Automated Pipeline:** End-to-end data pipeline from Google Drive integration to image preprocessing.
* **Scientific Validation:** Includes Confusion Matrix and Classification Reports (Precision/Recall) to verify reliability.
* **Live Demo:** Features an interactive "Upload Button" to test the model on real-world images from the internet.

## 🛠️ Tech Stack
* **Core:** Python 3.10+
* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab (TPU/GPU acceleration recommended)

## 📂 Dataset
The model was trained on the **UCI Ovarian Ultrasound Image Dataset**, containing approximately 6,800 images.
* **Source:** [Kaggle - UCI Ovarian Ultrasound Image Dataset](https://www.kaggle.com/datasets/ucimachinelearning/ovarian-ultrasound-image-dataset)
* **Classes:**
    1.  **Healthy:** Normal ovarian tissue.
    2.  **Dominant Follicle:** Normal physiological cyst (part of ovulation).
    3.  **Simple Cyst:** Fluid-filled, benign sac.
    4.  **Complex Cyst:** Solid/Irregular mass (potentially pathological).
    5.  **PCOS:** Polycystic Ovary Syndrome (multiple small follicles).

## 🧠 Model Architecture
We utilized a custom **CNN (Convolutional Neural Network)** designed for medical image feature extraction:
1.  **Input Layer:** Rescaling (0-1) and Resizing (224x224).
2.  **Convolutional Blocks:** Three blocks of `Conv2D` (32, 64, 128 filters) + `MaxPooling2D` to extract textures (like the "string of pearls" in PCOS).
3.  **Dense Layers:** A fully connected layer (128 units) with ReLU activation.
4.  **Output Layer:** Softmax activation for 5-class probability distribution.

## 📊 Results
* **Validation Accuracy:** 99.56%
* **Loss:** <0.05
* **Evaluation:** The Confusion Matrix demonstrates minimal error between "Simple Cysts" and "Dominant Follicles," proving the model's ability to discern subtle visual differences.

## 💻 How to Run
This project is optimized for **Google Colab**.

1.  **Clone/Download** this repository.
2.  **Download Data:** Get the `archive.zip` from the dataset link above.
3.  **Setup Drive:** Upload `archive.zip` to your Google Drive in a folder named `My_Ovary_Project`.
4.  **Open Notebook:** Open `Automated_Detection_of_PCOS_and_Ovarian_Cysts_using_CNNs.ipynb` in Google Colab.
5.  **Run All Cells:**
    * **Phases 1-4:** Will load data, build, and train the model (approx. 15 mins).
    * **Phase 5-6:** Will display accuracy charts and visual sanity checks.
    * **Phase 8:** Will launch the upload tool for you to test your own images.

## 🔮 Future Scope
* **Grad-CAM Integration:** To visualize *where* the model is looking (Heatmaps).
* **Mobile Deployment:** Converting the model to TensorFlow Lite for a mobile app.
* **Segmentation:** Drawing bounding boxes around the specific cysts.

## 📝 License
This project is for educational and research purposes.
Dataset Citation: UCI Machine Learning Repository.
