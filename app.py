
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import types
import torch.nn.functional as F
import boto3
import os
from pathlib import Path

# =====================================================================
# AWS S3 Configuration
# =====================================================================
S3_BUCKET_NAME = "your-bucket-name"  # Replace with your S3 bucket name
S3_MODEL_KEY = "models/model_for_inference.pth"  # Path to model in S3
LOCAL_MODEL_PATH = "/tmp/model_for_inference.pth"  # Temporary local path

# =====================================================================
# Download Model from S3
# =====================================================================
@st.cache_resource
def download_model_from_s3():
    """Download model from S3 if not already cached"""
    try:
        # Check if model already exists locally
        if os.path.exists(LOCAL_MODEL_PATH):
            st.info("Using cached model file")
            return LOCAL_MODEL_PATH
        
        st.info(f"Downloading model from S3 bucket: {S3_BUCKET_NAME}")
        
        # Initialize S3 client (uses IAM role credentials automatically on EC2)
        s3_client = boto3.client('s3')
        
        # Download model file
        s3_client.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        
        st.success(f"✅ Model downloaded successfully from S3")
        return LOCAL_MODEL_PATH
        
    except Exception as e:
        st.error(f"❌ Error downloading model from S3: {e}")
        st.info("""
        **Troubleshooting:**
        - Verify S3 bucket name and model path
        - Check EC2 instance has IAM role with S3 read permissions
        - Ensure model file exists in S3 bucket
        """)
        return None

# =====================================================================
# Grad-CAM for DenseNet
# =====================================================================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Hook the last convolutional layer of DenseNet
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # clone() fixes in-place modification issues
        self.activations = output.detach().clone()

    def save_gradient(self, module, grad_input, grad_output):
        # clone() prevents autograd view modification issue
        self.gradients = grad_output[0].detach().clone()

    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], device=input_tensor.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class, output


# =====================================================================
# Load Model
# =====================================================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download model from S3
    model_path = download_model_from_s3()
    if model_path is None:
        return None, None, None

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Create DenseNet121 model
        model = models.densenet121(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, checkpoint["num_classes"])

        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model = model.to(device)
        model.eval()

        # Patch: Disable in-place ReLU to fix Grad-CAM backward error
        def densenet_forward_no_inplace(self, x):
            features = self.features(x)
            out = F.relu(features.clone(), inplace=False)
            out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
            out = self.classifier(out)
            return out

        model.forward = types.MethodType(densenet_forward_no_inplace, model)

        return model, checkpoint["class_names"], device
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


# =====================================================================
# Image Preprocessing
# =====================================================================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def denormalize_image(tensor):
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    return img


def create_overlay(image_np, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = heatmap * 0.4 + image_np * 0.6
    return overlay


# =====================================================================
# Streamlit App
# =====================================================================
def main():
    st.set_page_config(page_title="DenseNet Image Classifier with Grad-CAM", layout="wide")

    st.title("DenseNet121 Image Classification with Grad-CAM")
    st.markdown("Upload an image to get predictions and visual explanations.")

    # Display environment info
    with st.expander("System Information"):
        st.write(f"**Device:** {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        st.write(f"**S3 Bucket:** {S3_BUCKET_NAME}")
        st.write(f"**Model Path:** {S3_MODEL_KEY}")

    try:
        model, class_names, device = load_model()
        
        if model is None:
            st.error("Failed to load model. Please check S3 configuration.")
            return
            
        st.success(f"Model loaded successfully! Classes: {', '.join(class_names)}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Ensure S3 bucket and IAM permissions are configured correctly.")
        return

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        input_tensor = preprocess_image(image).to(device)

        with st.spinner("Generating Grad-CAM explanation..."):
            gradcam = GradCAM(model)
            cam, pred_class, output = gradcam.generate_cam(input_tensor)

            probs = torch.nn.functional.softmax(output, dim=1)[0]
            top3_prob, top3_idx = torch.topk(probs, min(3, len(class_names)))

        with col2:
            st.subheader("Grad-CAM Heatmap")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(cam, cmap="jet")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()

        with col3:
            st.subheader("Explanation Overlay")
            img_display = denormalize_image(input_tensor)
            overlay = create_overlay(img_display, cam)
            st.image(overlay, use_container_width=True)

        st.markdown("---")
        st.subheader("Predictions")

        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            st.markdown("### Main Prediction")
            st.success(f"**{class_names[pred_class]}**")
            st.metric("Confidence", f"{probs[pred_class].item():.1%}")

        with col_pred2:
            st.markdown("### Top 3 Predictions")
            for prob, idx in zip(top3_prob, top3_idx):
                st.write(f"**{class_names[idx]}**: {prob.item():.1%}")

        st.markdown("---")
        st.info("""
        **Interpretation Guide (Grad-CAM):**
        - 🔴 Red → High importance
        - 🟡 Yellow → Moderate importance
        - 🔵 Blue → Low importance
        """)


if __name__ == "__main__":
    main()