import io
import os
from typing import Tuple

import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F


st.set_page_config(
    page_title="Brain MRI Tumor Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_model = nn.Sequential(
            nn.Linear(64 * 32 * 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x

@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load("cnn_brain_tumor_model.pth", map_location=device))
    model.eval()
    return model, device


def preprocess_image(img: Image.Image, target_size=(128, 128)) -> torch.Tensor:
    img = img.convert("RGB")
    img = img.resize(target_size)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = (img - 0.5) / 0.5 
    img = np.transpose(img, (2, 0, 1))  
    img_tensor = torch.tensor(img).unsqueeze(0)  
    return img_tensor

def predict_image(model: nn.Module, device, image_tensor: torch.Tensor):
    """Perform inference and return class label + probability."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()  
    return  prob


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


def render_prediction_card(prob_tumor: float):
    has_tumor = prob_tumor >= 0.5
    label = "Tumor Detected" if has_tumor else "No Tumor Detected"
    confidence_percent = prob_tumor if prob_tumor >= 0.5 else (1 - prob_tumor)
    score_text = f"Confidence: {confidence_percent*100:.1f}%"
    card_bg = "#ffe6e6" if has_tumor else "#e6ffe9"
    card_border = "#ff4d4f" if has_tumor else "#52c41a"
    card_emoji = "🚨" if has_tumor else "✅"

    st.markdown(
        f"""
        <div style="background:{card_bg};border:1px solid {card_border};padding:18px;border-radius:10px;">
            <div style="font-size:20px;font-weight:600;color:#222;display:flex;align-items:center;gap:8px;">
                <span>{card_emoji}</span>
                <span>{label}</span>
            </div>
            <div style="margin-top:6px;color:#555;">{score_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title("🧠 Brain MRI Tumor Detector")
    st.caption("Upload a brain MRI image to detect tumor presence.")

    with st.expander("Instructions", expanded=False):
        st.markdown(
            "- Upload a brain MRI image (JPG/PNG)\n"
            "- Click **Submit** to run the trained model\n"
            "- The prediction card will display the result"
        )

    uploaded = st.file_uploader(
        "Drag & drop a brain MRI image or click to browse",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Supported formats: JPG, JPEG, PNG",
    )

    submitted = st.button("Submit", type="primary", use_container_width=True, disabled=uploaded is None)

    if submitted and uploaded is not None:
        try:
            img = Image.open(io.BytesIO(uploaded.read()))
        except Exception as e:
            st.error(f"Could not read the image: {e}")
            return

        st.image(img, caption="Uploaded MRI", use_column_width=True)

        with st.spinner("Running inference..."):
            model, device = load_model()
            image_tensor = preprocess_image(img)
            prob = predict_image(model, device, image_tensor)
            
        render_prediction_card(prob)

    st.markdown("<small>Model: CNN trained on Brain MRI Dataset (128×128)</small>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()
