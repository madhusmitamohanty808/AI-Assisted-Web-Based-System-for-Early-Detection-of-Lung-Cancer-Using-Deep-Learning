# 🩺 AI-Assisted Web Based System for Early Detection of Lung Cancer  

## 📌 Project Overview  
This project is an **AI-powered web application** designed for the **early detection of lung cancer** from CT scan images.  
The system classifies CT scans into three categories:  

- **Normal** (healthy lungs)  
- **Benign** (non-cancerous tumor)  
- **Malignant** (cancerous tumor)  

The solution integrates **deep learning models** with **GAN-based synthetic image augmentation** to overcome dataset imbalance. The final trained model is deployed as an interactive **Streamlit web application**, allowing clinicians and researchers to upload CT scans and receive **real-time predictions**.  

---

## 🎯 Objectives  
- Detect lung cancer at an **early stage** using AI.  
- Reduce dependency on manual diagnosis and **minimize human errors**.  
- Create a **balanced and robust dataset** using **DCGAN augmentation**.  
- Deploy an **efficient and lightweight AI model** that works in real-time.  
- Provide an **accessible web-based diagnostic tool** for healthcare use.  

---

## ⚙️ Features  
✅ Multi-class classification: Normal, Benign, Malignant  
✅ GAN-based augmentation for dataset balancing  
✅ Comparative study of **4 deep learning models**  
✅ Optimized model: **MobileNetV2** (best accuracy vs. efficiency trade-off)  
✅ **Real-time predictions** via Streamlit web app  
✅ Outputs **confidence scores** for predictions  
✅ Easy deployment in **resource-constrained environments**  

---

## 🧠 Methodology  

### 🔹 1. Data Pre-processing  
- All images resized to **224×224**  
- Image normalization using ImageNet mean & std  
- Data split: **80% training, 10% validation, 10% testing**  
- Applied augmentations:  
  - Random affine transformations  
  - Random cropping  
  - Color jittering  

### 🔹 2. GAN-Based Augmentation  
- Implemented **Deep Convolutional GAN (DCGAN)**  
- Generated **5,000 synthetic CT images** (benign, malignant, normal)  
- Solved class imbalance problem and improved generalization  

### 🔹 3. Deep Learning Models  
Evaluated **four pre-trained models** using transfer learning:  
1. **ResNet50** – High accuracy but long training time  
2. **DenseNet121** – Strong feature reuse, fewer parameters  
3. **Vision Transformer (ViT)** – Transformer-based, powerful but computationally heavy  
4. **MobileNetV2** – Lightweight and efficient, chosen as final deployed model  

Each model’s last classification layer was replaced with:  
- Dropout (rate = 0.5)  
- Fully connected softmax classifier with **3 output classes**  

### 🔹 4. Evaluation Metrics  
- Accuracy  
- Precision, Recall, F1-Score  
- Training time & Inference time  
- Confusion Matrix  
- ROC-AUC  

### 🔹 5. Deployment  
- **Best model (MobileNetV2)** deployed using **Streamlit**  
- Web interface enables clinicians to:  
  - Upload CT scan images  
  - Get prediction (Normal / Benign / Malignant)  
  - View confidence percentages  

---

## 📊 Results  

### 🔹 Performance Comparison  

| Model        | Accuracy | F1-Score | Training Time | Inference Time |
|--------------|----------|----------|---------------|----------------|
| ResNet50     | 98.04%   | 0.9804   | 1315 min      | ~6.4 sec       |
| DenseNet121  | 97.88%   | 0.9780   | 929 min       | ~2.9 sec       |
| Vision Transformer (ViT) | 97.39% | 0.9740 | 1330 min | ~7.7 sec |
| **MobileNetV2** | **97.55%** | **0.9757** | **192 min** | **~1 sec** |

✅ **Conclusion:**  
- **ResNet50** → Best accuracy but very slow  
- **MobileNetV2** → Best balance of accuracy & speed (final deployed model)  

---

## 🛠️ Tech Stack  

- **Programming Language:** Python  
- **Frameworks:** TensorFlow, PyTorch  
- **Web Deployment:** Streamlit  
- **Libraries:** NumPy, OpenCV, Matplotlib, Scikit-learn  
- **GAN:** DCGAN for synthetic data generation  

---

## 📂 Project Structure  

```
├── data/                # Dataset (original + GAN-augmented images)
├── models/              # Trained deep learning models
├── notebooks/           # Jupyter notebooks for experiments
├── app/                 # Streamlit web app files
│   ├── app.py           # Main Streamlit app
├── requirements.txt     # Project dependencies
└── README.md            # Documentation
```

---

## ▶️ Installation & Usage  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/lung-cancer-detection.git
cd lung-cancer-detection
```

### 2️⃣ Create virtual environment (optional but recommended)  
```bash
python -m venv venv
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app  
```bash
streamlit run app/app.py
```

### 5️⃣ Upload CT Scan Image  
- The app will classify as **Normal / Benign / Malignant**  
- Displays prediction along with **confidence scores**  

---

## 📈 Sample Outputs  
- Uploading a **malignant CT scan** → Prediction: Malignant (Confidence: 99.8%)  
- Uploading a **normal CT scan** → Prediction: Normal (Confidence: 99.6%)  

---

## 🚀 Future Work  
- Explore **hybrid CNN + Transformer architectures** for higher accuracy  
- Deploy model to **edge devices / mobile apps**  
- Extend system to detect other **thoracic diseases** (e.g., pneumonia, TB)  
- Improve interpretability with **heatmaps / Grad-CAM visualization**  

---

## 🤝 Contribution  
Contributions are welcome!  
If you’d like to improve the project, feel free to fork the repo, make changes, and submit a pull request.  

---

## 📜 License  
This project is released under the **MIT License**.  

---

✨ This project demonstrates how **AI + Deep Learning** can enable **real-time, efficient, and accessible lung cancer detection tools**, bridging the gap between research and healthcare applications.  
