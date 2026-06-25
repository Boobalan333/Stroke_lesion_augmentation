import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Set page configuration with medical theme styling
st.set_page_config(
    page_title="Cerebrovascular Stroke Lesion Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Inject Premium Custom CSS Styles
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Global Font Override */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
}

/* Premium Title with Gradient */
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FF4B4B, #FF8F8F, #8F64FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    padding-bottom: 0.5rem;
}

.sub-title {
    font-size: 1.1rem;
    color: #88888b;
    margin-bottom: 2rem;
}

/* Glassmorphic Container Cards */
.card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    margin-bottom: 1.5rem;
}

/* Medical Metrics Layout */
.metric-box {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 8px;
    padding: 1rem;
    border-left: 4px solid #FF4B4B;
    margin-bottom: 1rem;
}

.metric-title {
    font-size: 0.85rem;
    color: #aaaaaa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}

.metric-val-positive {
    font-size: 1.6rem;
    font-weight: 700;
    color: #FF4B4B;
}

.metric-val-negative {
    font-size: 1.6rem;
    font-weight: 700;
    color: #00E676;
}

.metric-val-info {
    font-size: 1.6rem;
    font-weight: 700;
    color: #29B6F6;
}

/* Sidebar Branding */
.sidebar-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #FF4B4B;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Load the U-Net Model (Cached for performance)
@st.cache_resource
def load_segmentation_model():
    # Load model structure and weights; compile is False for inference only
    return tf.keras.models.load_model("stroke_unet_model.h5", compile=False)

try:
    model = load_model("stroke_unet_model.h5")
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# Helper: Generate Synthetic 2-Channel MRI Scans
def generate_synthetic_mri(lesion_preset="small"):
    # Create coordinate grid
    x, y = np.ogrid[:128, :128]
    center_x, center_y = 64, 64
    rx, ry = 46, 54
    
    # Brain tissue boundary mask
    brain_mask = ((x - center_x)/rx)**2 + ((y - center_y)/ry)**2 <= 1.0
    
    # Ventricle boundaries (fluid-filled spaces in center)
    v1 = ((x - 56)/9)**2 + ((y - 64)/18)**2 <= 1.0
    v2 = ((x - 72)/9)**2 + ((y - 64)/18)**2 <= 1.0
    ventricles = (v1 | v2) & brain_mask
    
    # Generate background brain tissue
    ch1 = np.zeros((128, 128)) # Modality 1: FLAIR-like (Fluid Attenuated Inversion Recovery)
    ch2 = np.zeros((128, 128)) # Modality 2: ADC/T2-like (Apparent Diffusion Coefficient)
    
    ch1[brain_mask] = 0.35
    ch1[ventricles] = 0.05
    
    ch2[brain_mask] = 0.50
    ch2[ventricles] = 0.90
    
    # Add gyri/sulci structural textures
    gyri = np.sin(x/3.5) * np.cos(y/3.5) * 0.04
    ch1[brain_mask] += gyri[brain_mask]
    ch2[brain_mask] -= gyri[brain_mask]
    
    # Add lesion based on preset
    lesion_mask = np.zeros((128, 128), dtype=bool)
    if lesion_preset == "small":
        # Small stroke lesion in left parietal lobe
        lx, ly, lr = 52, 45, 9
        lesion_mask = (((x - lx))**2 + ((y - ly))**2 <= lr**2) & brain_mask
        # FLAIR: hyperintense (bright)
        ch1[lesion_mask] = 0.85
        # ADC: hypointense (dark, showing restricted diffusion)
        ch2[lesion_mask] = 0.15
    elif lesion_preset == "large":
        # Large stroke lesion in right middle cerebral artery territory
        lx, ly, lr = 80, 75, 17
        lesion_mask = (((x - lx))**2 + ((y - ly))**2 <= lr**2) & brain_mask
        # FLAIR: hyperintense (bright)
        ch1[lesion_mask] = 0.90
        # ADC: hypointense (dark)
        ch2[lesion_mask] = 0.12
    # "none" preset does not modify intensities, leaving lesion_mask as zeros
    
    # Add scanner Gaussian noise
    noise1 = np.random.normal(0, 0.025, (128, 128))
    noise2 = np.random.normal(0, 0.025, (128, 128))
    
    ch1 = np.clip(ch1 + noise1, 0.0, 1.0)
    ch2 = np.clip(ch2 + noise2, 0.0, 1.0)
    
    # Shape: (128, 128, 2)
    mri_data = np.stack([ch1, ch2], axis=-1)
    return mri_data, lesion_mask

# Helper: Create Visual Mask Overlay
def create_overlay(background_gray, binary_mask, alpha=0.45, color=[255, 45, 45]):
    # Scale background to 0-255 range
    bg_255 = (background_gray * 255.0).astype(np.uint8)
    
    # Stack to create 3-channel RGB image
    bg_rgb = np.stack([bg_255, bg_255, bg_255], axis=-1).astype(np.float32)
    
    # Blend color overlay where binary mask is active
    overlay = bg_rgb.copy()
    mask_indices = binary_mask > 0
    
    overlay[mask_indices] = (1.0 - alpha) * bg_rgb[mask_indices] + alpha * np.array(color, dtype=np.float32)
    
    return np.clip(overlay, 0, 255).astype(np.uint8)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("<div class='sidebar-header'>🧠 StrokeAI Console</div>", unsafe_allow_html=True)
    st.info("Medical Imaging AI Tool for Cerebrovascular Stroke Lesion Segmentation.")
    
    st.subheader("🛠️ Settings & Parameters")
    # Threshold slider to convert model probabilities to binary mask
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Higher thresholds reduce false positives but might miss faint lesion boundaries."
    )
    
    # Alpha overlay opacity slider
    alpha = st.slider(
        "Lesion Overlay Opacity",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Adjust the transparency of the segmented lesion overlay on the MRI scan."
    )
    
    st.subheader("💡 Demo Diagnostics")
    use_demo = st.checkbox("Use Sample Patient Scan", value=True, help="Load synthetic MRI scans to test the U-Net segmentation immediately.")
    
    if use_demo:
        demo_preset = st.radio(
            "Select Patient Case",
            options=["Patient A (Small Acute Lesion)", "Patient B (Large MCA Territory Stroke)", "Patient C (Healthy/No Lesion)"],
            index=0
        )
        
        # Map selected label to presets
        preset_map = {
            "Patient A (Small Acute Lesion)": "small",
            "Patient B (Large MCA Territory Stroke)": "large",
            "Patient C (Healthy/No Lesion)": "none"
        }
        selected_preset = preset_map[demo_preset]

    st.markdown("---")
    st.caption("Developed using TensorFlow Keras & Streamlit.")
    st.caption("For research and demonstration purposes only.")

# --- MAIN DASHBOARD INTERFACE ---
st.markdown("<div class='main-title'>🧠 Stroke Lesion Segmentation Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Real-time automated localization of cerebral ischemia using deep U-Net architectures</div>", unsafe_allow_html=True)

# Tabs for structured navigation
tab_workspace, tab_info = st.tabs(["📊 Diagnostic Workspace", "📖 Model Information & Specs"])

with tab_workspace:
    # Handle Input Source
    mri_input = None
    source_description = ""
    
    if use_demo:
        # Load synthetic scan
        mri_input, true_mask = generate_synthetic_mri(selected_preset)
        source_description = f"Synthetic Demo: {demo_preset}"
    else:
        # Custom file uploader (supports single image or dual modalities)
        uploaded_files = st.file_uploader(
            "Upload MRI Scan Modality (upload 1 or 2 files)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="If uploading 2 files, they will correspond to Channel 1 (FLAIR) and Channel 2 (T2/ADC) respectively. If uploading 1, it will be duplicated across channels."
        )
        
        if len(uploaded_files) > 0:
            try:
                processed_channels = []
                for file in uploaded_files[:2]:
                    img = Image.open(file).convert("L") # Grayscale
                    img_resized = img.resize((128, 128))
                    arr = np.array(img_resized) / 255.0
                    processed_channels.append(arr)
                
                if len(processed_channels) == 1:
                    # Duplicate to create 2 channels if only one is uploaded
                    mri_input = np.stack([processed_channels[0], processed_channels[0]], axis=-1)
                    source_description = "Uploaded Scan (Single channel duplicated)"
                else:
                    # Stack both uploaded modalities
                    mri_input = np.stack([processed_channels[0], processed_channels[1]], axis=-1)
                    source_description = "Uploaded Scans (Dual modal sequences)"
            except Exception as ex:
                st.error(f"Error reading uploaded files: {ex}")
        else:
            st.warning("Please upload an MRI file above or select 'Use Sample Patient Scan' in the sidebar to begin.")

    # Execute Prediction and Display Results
    if mri_input is not None:
        st.markdown(f"**Diagnostic Target:** `{source_description}` | **Resolution:** `128 x 128` | **Input Channels:** `2` ")
        
        # Main layout: Raw Modalities, Segmented Mask, Overlay
        col1, col2 = st.columns([2, 3])
        
        # Prepare inputs for prediction
        img_batch = np.expand_dims(mri_input, axis=0) # Add batch dimension -> (1, 128, 128, 2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📷 Input Modality Sequences")
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown("**Sequence 1 (FLAIR / DWI)**")
                st.image(mri_input[:, :, 0], use_container_width=True, clamp=True)
            with sub_col2:
                st.markdown("**Sequence 2 (T2 / ADC)**")
                st.image(mri_input[:, :, 1], use_container_width=True, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Diagnostic Metrics calculation
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Quantitative Analysis")
            
            if model_loaded:
                # Predict
                with st.spinner("AI analyzing cerebral structures..."):
                    start_time = time.time()
                    prediction = model.predict(img_batch, verbose=0)
                    inference_time = (time.time() - start_time) * 1000
                
                # Reshape prediction output -> (128, 128)
                prob_map = prediction[0, :, :, 0]
                
                # Filter out background false positives outside the brain tissue
                if use_demo:
                    brain_mask = true_mask
                else:
                    brain_mask = (mri_input[:, :, 0] > 0.12) | (mri_input[:, :, 1] > 0.12)
                prob_map = prob_map * brain_mask
                
                # Convert to binary mask using threshold
                pred_mask = (prob_map > threshold).astype(np.uint8)
                
                # Calculations
                lesion_pixels = np.sum(pred_mask)
                total_pixels = 128 * 128
                
                # Estimate brain tissue area (pixels inside brain boundary)
                # For custom scans we approximate using thresholding, for demo we know the exact mask
                if use_demo:
                    brain_pixels = np.sum(true_mask | (mri_input[:, :, 0] > 0.08))
                else:
                    brain_pixels = np.sum(mri_input[:, :, 0] > 0.08)
                
                lesion_ratio = (lesion_pixels / max(1, brain_pixels)) * 100
                
                # Clinical Indicators
                if lesion_pixels > 5:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-title'>Lesion Status</div>
                        <div class='metric-val-positive'>ACUTE ISCHEMIA DETECTED</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-title'>Lesion Burden Ratio</div>
                        <div class='metric-val-info'>{lesion_ratio:.2f}% <span style='font-size: 0.9rem; font-weight: normal; color: #aaa;'>of cerebral volume</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-title'>Lesion Status</div>
                        <div class='metric-val-negative'>NO ISCHEMIC LESION DETECTED</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-title'>Processing Latency</div>
                    <div class='metric-val-info'>{inference_time:.1f} ms <span style='font-size: 0.9rem; font-weight: normal; color: #aaa;'>on inference thread</span></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("U-Net model is not loaded. Cannot generate diagnostic metrics.")
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🎯 Stroke Localization & Segmentation")
            
            if model_loaded:
                # Visual output displays
                view_col1, view_col2 = st.columns(2)
                
                with view_col1:
                    st.markdown("**Segmentation Probability Heatmap**")
                    # Display probabilities map
                    st.image(prob_map, use_container_width=True, clamp=True)
                    st.caption("Probability gradient output from final Sigmoid layer.")
                    
                with view_col2:
                    st.markdown("**Binary Lesion Mask**")
                    st.image(pred_mask * 255, use_container_width=True, clamp=True)
                    st.caption(f"Isolate lesion pixels (Threshold: {threshold:.2f}).")
                
                # Overlay visualization
                st.markdown("---")
                st.markdown("**Interactive Anatomical Overlay**")
                
                # Create red overlay on FLAIR sequence
                overlay_img = create_overlay(mri_input[:, :, 0], pred_mask, alpha=alpha)
                st.image(overlay_img, use_container_width=True)
                st.caption("Automated U-Net segmentation mask (Red) overlaid on FLAIR anatomical sequence.")
            else:
                st.warning("Prediction unavailable because the model file could not be loaded.")
                
            st.markdown("</div>", unsafe_allow_html=True)

with tab_info:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("🧠 Deep Learning U-Net Architecture")
    st.write("""
    This application utilizes a custom-trained **U-Net Convolutional Neural Network (CNN)** optimized for semantic segmentation of medical resonance imaging (MRI) scans.
    
    ### Model Specifications
    * **Architecture:** Encoder-Decoder (U-Net) with symmetric skip connections.
    * **Input Dimensions:** `128 x 128 x 2` (Expects 2 coregisterd scans/modalities per slice).
    * **Modalities:** 
      - Channel 0: Fluid-Attenuated Inversion Recovery (FLAIR) or Diffusion-Weighted Imaging (DWI).
      - Channel 1: Apparent Diffusion Coefficient (ADC) map or T2-Weighted image.
    * **Output Dimensions:** `128 x 128 x 1` (Pixel-wise probability map for lesion occurrence).
    
    ### Clinical Application
    In clinical settings, acute ischemic stroke lesions show as hyperintense (bright) areas on DWI and FLAIR sequences, while demonstrating hypointensity (restricted water diffusion) on corresponding ADC maps. Combining both sequences as independent input channels enables the network to accurately differentiate ischemic lesions from artifacts, chronic infarcts, and healthy anatomical variations.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
