import os
import time
import logging
import streamlit as st
import requests
import torch
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, CLIPModel, AutoProcessor
from PIL import Image

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

# # Ensure Hugging Face authentication
# from huggingface_hub import login
# login(HF_ACCESS_TOKEN)

# Load CLIP model and processor
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure the index exists
index_name = "index-search"
if not pc.has_index(index_name):
    pc.create_index(name=index_name, metric="cosine",
                    dimension=512,
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    time.sleep(5)  # Wait for index to initialize

unsplash_index = pc.Index(index_name)

# Streamlit UI
st.title("Search Images by Text or Image")

search_mode = st.radio("Choose search mode:", ["Text Search", "Image Search"])

if search_mode == "Text Search":
    search_query = st.text_input("Search (at least 3 characters)")
    if len(search_query) >= 3:
        with st.spinner("Searching images..."):
            inputs = tokenizer([search_query], padding=True, return_tensors="pt")
            text_features = model.get_text_features(**inputs)
            text_embedding = text_features.detach().numpy().flatten().tolist()

            response = unsplash_index.query(
                top_k=10,
                vector=text_embedding,
                namespace="image-search-dataset",
                include_metadata=True
            )

        # Display results
        cols = st.columns(2)
        for i, result in enumerate(response.matches):
            with cols[i % 2]:
                st.image(result.metadata["url"], caption=f"Score: {result.score:.4f}")

elif search_mode == "Image Search":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Searching similar images..."):
            inputs = processor(images=image, return_tensors="pt")
            image_features = model.get_image_features(**inputs)
            image_embedding = image_features.detach().numpy().flatten().tolist()

            response = unsplash_index.query(
                top_k=10,
                vector=image_embedding,
                namespace="image-search-dataset",
                include_metadata=True
            )

        # Display results
        cols = st.columns(2)
        for i, result in enumerate(response.matches):
            with cols[i % 2]:
                st.image(result.metadata["url"], caption=f"Score: {result.score:.4f}")
