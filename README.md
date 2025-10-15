# 🧠 Brain Tumor Detection using CNN on MRI Scans

## Project Overview
This project implements a **Convolutional Neural Network (CNN)** to detect brain tumors from MRI scans. Developed with **PyTorch**, the model classifies tumor presence in MRI images. Key features include:

- **Data augmentation** for better generalization.
- **Streamlit app** for interactive predictions.
- End-to-end pipeline: data preprocessing → model training → evaluation → deployment.

---

## Features
- Custom **CNN architecture** for MRI classification.
- Preprocessing and **augmentation** (rotation, flipping, scaling).
- Split data into **training**, **validation**, and **testing** sets.
- PyTorch training with **loss calculation**, **accuracy metrics**, and **visualizations**.
- Exported model for inference.
- **Streamlit interface** for user-friendly predictions.

---

## Dataset
- Brain MRI images ([public dataset:](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/data))
- Classes:
  - `Tumor`
  - `No Tumor`

---

## Project Structure






---

## Installation
1. Clone the repository:
```bash
git clone git@github.com:Baya-Mezghani/Brain-MRI-Tumor-Detection-.git
cd brain-tumor-detection-
```
2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements1.txt
```

---


## Usage
Run Streamlit App 
```bash
streamlit run streamlit_app.py
```
---

## Results

### Model Performance
- **Accuracy:** 0.73  
- **Comments:** This is a solid baseline for a custom CNN with data augmentation. Further improvements are possible with transfer learning or advanced architectures.

---

### Example Predictions

**Streamlit Interface Screenshot:**  
![Streamlit App](screenshots/streamlit_app_screenshot.png)

**Sample Predictions:**

| MRI Image | Model Prediction |
|-----------|----------------|
| ![brain1](screenshots/brain1.jpg) | Tumor |
| ![brain2](screenshots/brain2.jpg) | No Tumor |

