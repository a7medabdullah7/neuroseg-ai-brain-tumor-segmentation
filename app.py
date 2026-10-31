
import io

import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from model import load_model, DEVICE


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="NeuroSeg AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0b1020;
        color: #f8fafc;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #a78bfa;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .result-card {
        background: #151d32;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #293451;
        margin-bottom: 12px;
    }

    .stButton > button {
        background-color: #7c3aed;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        padding: 10px 24px;
    }

    .stButton > button:hover {
        background-color: #6d28d9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load Model Once
# --------------------------------------------------
@st.cache_resource
def get_model():
    return load_model()


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------
def preprocess_image(image):
    image = image.convert("L")
    image = image.resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        image_array
    ).unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE), image_array


# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict(image, model, threshold):
    input_tensor, original_array = preprocess_image(image)

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits)
        prediction = (
            probabilities >= threshold
        ).float()

    probability_map = probabilities[0, 0].cpu().numpy()
    mask = prediction[0, 0].cpu().numpy().astype(np.uint8)

    return original_array, mask, probability_map


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">NeuroSeg AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Brain MRI Tumor Segmentation Platform | U-Net + ResNet34'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## NeuroSeg AI")
    st.caption("Medical Image Segmentation")

    st.markdown("---")

    st.markdown("### Model Information")
    st.write("Architecture: U-Net")
    st.write("Encoder: ResNet34")
    st.write("Input: Grayscale MRI")
    st.write("Image size: 256 × 256")
    st.write("Threshold: 0.60")

    st.markdown("---")

    st.markdown("### Evaluation Results")
    st.metric("Positive Dice", "72.71%")
    st.metric("Positive IoU", "62.13%")
    st.metric("Empty-mask FPR", "26.57%")

    st.markdown("---")
    st.caption(f"Running device: {DEVICE}")


# --------------------------------------------------
# Main Interface
# --------------------------------------------------
st.markdown("## Upload MRI Scan")

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("### Uploaded Image")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                image,
                caption="Original MRI",
                use_container_width=True
            )

        with col2:
            st.markdown(
                '<div class="result-card">'
                '<h4>Image Information</h4>'
                f'<p>Width: {image.width} px</p>'
                f'<p>Height: {image.height} px</p>'
                '<p>Format: MRI Image</p>'
                '</div>',
                unsafe_allow_html=True
            )

        if st.button(
            "Analyze MRI Scan",
            use_container_width=True
        ):

            with st.spinner("Analyzing MRI image..."):

                model, threshold = get_model()

                original, mask, probability_map = predict(
                    image,
                    model,
                    threshold
                )

            tumor_pixels = int(mask.sum())
            total_pixels = int(mask.size)
            tumor_percentage = (
                tumor_pixels / total_pixels
            ) * 100

            # Overlay
            fig, ax = plt.subplots(figsize=(7, 7))

            ax.imshow(original, cmap="gray")
            ax.imshow(
                np.ma.masked_where(mask == 0, mask),
                cmap="Reds",
                alpha=0.45,
                vmin=0,
                vmax=1
            )

            ax.axis("off")
            ax.set_title("MRI Tumor Segmentation")

            st.markdown("## Segmentation Results")

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.image(
                    original,
                    caption="Original MRI",
                    use_container_width=True
                )

            with result_col2:
                st.image(
                    mask * 255,
                    caption="Predicted Tumor Mask",
                    use_container_width=True
                )

            with result_col3:
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("## Prediction Summary")

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Predicted Tumor Pixels",
                    f"{tumor_pixels:,}"
                )

            with metric2:
                st.metric(
                    "Predicted Area",
                    f"{tumor_percentage:.2f}%"
                )

            # Download mask
            mask_image = Image.fromarray(
                mask * 255
            )

            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")

            st.download_button(
                "Download Segmentation Mask",
                data=buffer.getvalue(),
                file_name="brain_tumor_mask.png",
                mime="image/png",
                use_container_width=True
            )

            st.info(
                "This application is a research prototype "
                "and is not intended for clinical diagnosis."
            )

    except Exception as error:
        st.error(f"Processing error: {error}")

else:
    st.info(
        "Upload a brain MRI image to start segmentation."
    )

    st.markdown("""
    ### How to use
    1. Upload a brain MRI image.
    2. Click Analyze MRI Scan.
    3. Review the predicted segmentation.
    4. Download the generated mask.
    """)
