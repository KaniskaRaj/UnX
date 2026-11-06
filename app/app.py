import streamlit as st
import numpy as np
from PIL import Image

st.title("UnCXR: Uncertainty-Aware Chest X-Ray Diagnosis")
st.markdown("---")

# Mandatory Safety Disclaimer
st.warning("⚠️ **Research prototype, not for clinical use.** This tool is built for academic evaluation only.")

# File uploader
uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    
    st.subheader("Analysis & Predictions")
    st.info("Model predictions and uncertainty badges will appear here once connected to your team's .npy prediction files!")
    
    # Mocking dummy output for UI testing
    st.metric(label="Predicted Status", value="Normal / Low Risk", delta="-2.1% Uncertainty")
    
    # Uncertainty Badge Example
    uncertainty_level = "Medium"  # Can be Low, Medium, High based on thresholds
    if uncertainty_level == "High":
        st.error("Uncertainty Badge: HIGH 🔴 (Heatmap hidden due to high model uncertainty)")
    elif uncertainty_level == "Medium":
        st.warning("Uncertainty Badge: MEDIUM 🟡")
    else:
        st.success("Uncertainty Badge: LOW 🟢")
