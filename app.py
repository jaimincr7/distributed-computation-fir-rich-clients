import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1

# Initialize the model
@st.cache_resource
def load_model():
    model = InceptionResnetV1(pretrained='vggface2').eval()
    return model

model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

def compute_embedding(image):
    """Compute the embedding for a given image using the pre-trained model."""
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(image).cpu().numpy().flatten()
    return embedding

def compute_cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    return similarity

# Streamlit App Layout
st.title("Face Recognition")

st.write(
    "This project is dedicated on research regarding the Distributed"
    "Machine Computation for Facial Image Recognition for Rich Clients."
    "Upload two images to compute their cosine similarity using a "
    "custom build facial recognition model."
)

# Upload two images
image1_file = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"])
image2_file = st.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"])

if image1_file and image2_file:
    # Load images
    image1 = Image.open(image1_file).convert("RGB")
    image2 = Image.open(image2_file).convert("RGB")

    # Display images
    st.image([image1, image2], caption=["Image 1", "Image 2"], width=300)

    # Compute embeddings and similarity
    embedding1 = compute_embedding(image1)
    embedding2 = compute_embedding(image2)

    similarity = compute_cosine_similarity(embedding1, embedding2)

    # Display result
    st.write(f"**Cosine Similarity:** {similarity:.4f}")

    # Optional: Add threshold to classify as same/different
    threshold = st.slider("Set similarity threshold", 0.0, 1.0, 0.5)
    if similarity >= threshold:
        st.success("The two images are of the **same person**.")
    else:
        st.error("The two images are of **different persons**.")
