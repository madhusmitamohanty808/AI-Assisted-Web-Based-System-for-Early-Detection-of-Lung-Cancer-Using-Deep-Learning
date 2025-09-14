# 🫁 AI-Assisted Web Based System for Early Detection of Lung Cancer  

This repository contains the implementation of our research work:  
**"AI-Assisted Web Based System for Early Detection of Lung Cancer Using Deep Learning"**  

Lung cancer is one of the leading causes of cancer-related deaths worldwide.  
Early and accurate detection can significantly improve patient survival.  
This project leverages **Deep Learning + GAN-based data augmentation** to build a **Streamlit web application**  
for real-time classification of lung CT scans into **Normal, Benign, or Malignant**.

---

## 🚀 Features
- ✅ CT-scan based lung cancer detection  
- ✅ Handles **imbalanced datasets** using **DCGAN-generated synthetic images**  
- ✅ Evaluation of **4 state-of-the-art deep learning models**: ResNet50, DenseNet121, Vision Transformer (ViT), MobileNetV2  
- ✅ Achieves **97.55% accuracy** with MobileNetV2 (best speed-performance trade-off)  
- ✅ **Streamlit-based deployment** for real-time clinical use  
- ✅ Lightweight, portable, and efficient for **low-resource environments**  

---

## 📊 Dataset
We used the **IQ-OTH/NCCD Lung Cancer Dataset** (Kaggle):  
🔗 [Lung Cancer Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-cancer-dataset)

- Benign: 120 images  
- Malignant: 561 images  
- Normal: 416 images  

To address imbalance, we generated **5,000 synthetic CT images** using **DCGAN**:  
- Benign → 2,000  
- Malignant → 2,000  
- Normal → 1,000  

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Frameworks & Libraries:** PyTorch, TensorFlow, NumPy, OpenCV, Matplotlib  
- **Deep Learning Models:** ResNet50, DenseNet121, Vision Transformer (ViT), MobileNetV2  
- **GAN Augmentation:** DCGAN  
- **Deployment:** Streamlit  

---

## ⚙️ Installation
Clone this repository:
```bash
git clone https://github.com/your-username/lung-cancer-detection.git
cd lung-cancer-detection
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

Upload a CT scan image and get predictions (**Normal / Benign / Malignant**) with confidence scores.

---

## 📈 Results
| Model          | Accuracy | F1-score | Training Time | Inference Time |
|----------------|----------|----------|---------------|----------------|
| ResNet50       | 98.04%   | 0.9804   | 1315 min      | 6.42s          |
| DenseNet121    | 97.88%   | 0.978    | 929 min       | 2.96s          |
| Vision Transformer | 97.39% | 0.974  | 1330 min      | 7.76s          |
| **MobileNetV2**| **97.55%** | **0.9757** | **192 min**  | **0.99s**      |

📌 **MobileNetV2** was chosen for deployment due to its efficiency and speed.

---

## 🌐 Deployment
- Implemented using **Streamlit** for an interactive web-based interface.  
- Allows clinicians to upload CT scans and receive real-time predictions.  
- Optimized for **low-resource healthcare settings**.  

---

## 📌 Future Work
- Explore **hybrid deep learning models** for improved accuracy.  
- Extend system for **multi-disease detection** in medical imaging.  
- Integration with **hospital PACS systems** for seamless adoption.  

---

## ✍️ Authors
- [Your Name / Team Name]  
- Based on research work presented in **IEEE Paper**.  

---

## 📜 License
This project is licensed under the **MIT License** – feel free to use and modify with attribution.  
