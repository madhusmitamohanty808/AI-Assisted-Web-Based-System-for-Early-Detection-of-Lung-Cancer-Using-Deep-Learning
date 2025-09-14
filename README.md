# 🫁 AI-Assisted Web Based System for Early Detection of Lung Cancer Using Deep Learning

This project proposes an **AI-assisted, web-based SYSTEM for the early detection of lung cancer using CT scan images**.
By leveraging **deep learning models** and **synthetic data generation (DCGAN),** the system can classify lung CT images into three categories: **Benign, Malignant, and Normal.**
A user-friendly **web interface (Streamlit)** allows users to upload CT scan images and get instant prediction

---

🧾 **Abstract**

Lung cancer is among the leading causes of cancer-related deaths worldwide, where early detection is critical for effective treatment. Manual CT image analysis is time-consuming, error-prone, and requires expert radiologists.

This project addresses these challenges by:

Expanding a limited dataset using Deep Convolutional GAN (DCGAN)

Training multiple deep learning models (ResNet50, MobileNetV2, DenseNet121, Vision Transformer)

Selecting MobileNetV2 as the best-performing model due to its accuracy, speed, and efficiency

Deploying the final model via a web-based system for real-time medical use

## 🚀 Features
- ✅ CT-scan based lung cancer detection  
- ✅ Handles **imbalanced datasets** using **DCGAN-generated synthetic images**  
- ✅ Evaluation of **4 state-of-the-art deep learning models**: ResNet50, DenseNet121, Vision Transformer (ViT), MobileNetV2  
- ✅ Achieves **97.55% accuracy** with MobileNetV2 (best speed-performance trade-off)  
- ✅ **Streamlit-based deployment** for real-time clinical use  
- ✅ Lightweight, portable, and efficient for **low-resource environments**  

---

## 🚀 Key Features
- **Automated Diagnosis**: Classifies CT scan images into **Benign, Malignant, or Normal** categories.
- **Synthetic Data Generation**: Uses **DCGAN** to address dataset imbalance by generating additional training samples.
- **Deep Learning Models**: Trained and compared models including **ResNet50, MobileNetV2, DenseNet121, Vision Transformer (ViT)**.
- **Best Performing Model**: **MobileNetV2** achieved the highest accuracy and efficiency, suitable for real-time deployment.
- **Web Deployment**: Streamlit-based interface for real-time predictions with easy image uploads.

## 📊 Dataset
- **Original Data**:  
  - Benign: 120 images  
  - Malignant: 561 images  
  - Normal: 426 images  
- **Synthetic Data (DCGAN generated)**:  
  - Benign: +2000 images  
  - Malignant: +2000 images  
  - Normal: +1000 images  

Final dataset was **balanced and preprocessed** (resizing, normalization, augmentation).

## 🧠 Deep Learning Models & Results
- **ResNet50**: High accuracy but computationally expensive.  
- **DenseNet121**: Competitive but slower.  
- **Vision Transformer (ViT)**: Good performance, high computational cost.  
- **MobileNetV2**: Best trade-off with **97.55% accuracy**, **F1-score: 0.9757**, and fastest inference time (~0.99s).  

## 🌐 Web Deployment
- Implemented using **Streamlit**.  
- Allows users to **upload CT scan images** (`.jpg`, `.jpeg`, `.png`).  
- Provides **real-time predictions** with class labels and confidence scores.  

## 📦 Tech Stack
- **Python**
- **TensorFlow / PyTorch**
- **DCGAN** (for data augmentation)
- **MobileNetV2**
- **Streamlit** (for deployment)
- **NumPy, OpenCV, Scikit-learn, Matplotlib, Seaborn** (for preprocessing & visualization)

## 📈 Performance Highlights
- Accuracy: **97.55%**
- F1-Score: **0.9757**
- Fast inference (< 1s per image)
- Robust predictions with high confidence

## 📍 How to Run the Project
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/AI-Lung-Cancer-Detection.git
   cd AI-Lung-Cancer-Detection
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```

## 📌 Future Enhancements
- Integration with **cloud platforms** for large-scale deployment.  
- Extending dataset with **real clinical data**.  
- Implementing **explainable AI (XAI)** for better interpretability.  


