# Stroke Lesion Segmentation Dashboard

An interactive, medical-grade deep learning dashboard for real-time automated localization and segmentation of acute cerebral ischemia from MRI scan modalities using a custom-trained **U-Net** architecture.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)

---


---

##  Features

- **Dual-Modality Sequencing**: Expects a 2-channel coregistered input representing different MRI sequences (e.g., FLAIR/DWI for hyperintensity highlights, and ADC map to confirm restricted water diffusion).
- **Interactive Workspace**:
  - Upload custom MRI scans or select from pre-packaged clinical presets.
  - Dynamically tune the **Confidence Threshold** to adjust segmentation sensitivity.
  - Adjust the **Anatomical Overlay Opacity** with real-time feedback.
- **Quantitative Clinical Metrics**:
  - **Lesion Status**: Real-time ischemic warning indicator.
  - **Lesion Burden Ratio**: Percentage of brain tissue affected.
  - **Processing Latency**: Diagnostic execution speed (typically < 50ms on standard CPUs).
- **Premium Visualization**:
  - Glassmorphic container design with a custom medical dashboard theme.
  - Probability heatmaps alongside binary threshold masks.
  - Alpha-blended, color-coded segmentation overlay on top of original anatomical sequences.

---

## Model Specifications & Architecture

The system features a deep encoder-decoder **U-Net** model designed for medical image segmentation:

| Parameter | Specification | Description |
| :--- | :--- | :--- |
| **Model Weight File** | `stroke_unet_model.h5` | Saved TensorFlow/Keras model structure & weights. |
| **Input Shape** | `(128, 128, 2)` | Height × Width × Modality Channels. |
| **Channel 0** | FLAIR / DWI | Highlights acute cytotoxic edema as bright regions. |
| **Channel 1** | ADC / T2 | Confirms ischemic core via hypointensity (restricted diffusion). |
| **Output Shape** | `(128, 128, 1)` | Pixel-wise probability map (Sigmoid activation). |

---

##  Project Structure

```
stroke-lesion-augmentation/
├── app.py                     # Streamlit application containing UI logic and synthetic generators
├── stroke_unet_model.h5       # Pre-trained U-Net weights & architecture
└── README.md                  # Project documentation (this file)
```

---

##  Installation & Getting Started

### Prerequisites
Make sure you have **Python 3.8+** installed on your machine.

### 1. Clone or Open the Project Directory
Navigate to the directory containing the code:
```bash
cd "path/to/stroke lesion augmentation"
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
Create and activate a Python virtual environment:
```bash
# Create environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required scientific computing, deep learning, and web dashboard libraries:
```bash
pip install streamlit tensorflow numpy pillow
```

### 4. Launch the Dashboard
Run the Streamlit server local instance:
```bash
streamlit run app.py
```
After execution, the app will open automatically in your browser (usually at `http://localhost:8501`).

---

##  Clinical Presets (Demo Mode)

For validation without live MRI data, the application includes three synthetic clinical presets:
1. **Patient A (Small Acute Lesion)**: Replicates a small acute ischemic stroke in the left parietal lobe with hyperintense FLAIR and hypointense ADC map.
2. **Patient B (Large MCA Territory Stroke)**: Simulates a massive right Middle Cerebral Artery (MCA) infarct spanning multiple cerebral zones.
3. **Patient C (Healthy / No Lesion)**: Simulates a normal cerebral structure to verify model specificity and ensure zero false positives.

---

