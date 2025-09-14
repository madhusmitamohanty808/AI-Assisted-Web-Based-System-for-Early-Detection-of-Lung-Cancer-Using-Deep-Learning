import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained MobileNetV2 model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[1].in_features

# Modify the classifier
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, 3)  # 3 classes: Benign, Malignant, Normal
)

# Load trained weights
model.load_state_dict(torch.load("best_mobilenetv2_model.pth", map_location=device))
model.to(device)
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels
class_names = ["Benign", "Malignant", "Normal"]

# Streamlit app interface
st.title("Lung Cancer Classification with MobileNetV2")
st.write("Upload a CT scan image to classify it into one of the following categories: Benign, Malignant, or Normal.")

# Upload file
uploaded_file = st.file_uploader("Choose a CT scan image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
        predicted_class = class_names[np.argmax(probabilities)]

    st.subheader("Prediction")
    st.write(f"**Predicted Class:** {predicted_class}")

    st.subheader("Confidence Scores")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {probabilities[i] * 100:.2f}%")

    fig, ax = plt.subplots()
    ax.bar(class_names, probabilities * 100, color=['blue', 'red', 'green'])
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Confidence Scores for Each Class")
    st.pyplot(fig)

st.markdown("---")
st.write("Developed with Streamlit and PyTorch")
