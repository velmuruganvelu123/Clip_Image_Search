# Add src directory to path
src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.append(src_directory)
import os
import sys
import logging
from transformers import AutoProcessor, CLIPModel
from database import create_pinecone_index
from data import request_method
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)



# Set Hugging Face token
load_dotenv()
HF_ACCESS_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_data):
    """
    Processes an image, generates embeddings using CLIP, and indexes it in Pinecone.
    
    Args:
        image_data (dict): A dictionary containing 'photo_id' and 'photo_image_url'.
    
    Returns:
        str: Success or error message.
    """
    try:
        if not isinstance(image_data, dict):
            raise ValueError("Invalid input: Expected a dictionary with 'photo_id' and 'photo_image_url'")

        photo_id = image_data.get("photo_id")
        url = image_data.get("photo_image_url")
        
        if not photo_id or not url:
            raise ValueError("Missing 'photo_id' or 'photo_image_url' in input data")
        
        image = request_method.get_urlimage(image_data)
        inputs = processor(images=image, return_tensors="pt")
        image_features = model.get_image_features(**inputs)
        embeddings = image_features.detach().cpu().numpy().flatten().tolist()
        
        pinecone_index = create_pinecone_index.get_index()
        pinecone_index.upsert(
            vectors=[
                {
                    "id": photo_id,
                    "values": embeddings,
                    "metadata": {
                        "url": url,
                        "photo_id": photo_id
                    }
                },
            ],
            namespace="image-search-dataset"
        )
        
        logger.info(f"Successfully indexed image {photo_id}")
        return f"Successfully indexed image {photo_id}"
    
    except Exception as e:
        logger.error(f"Error processing image {image_data}: {e}")
        return f"Error processing image {photo_id}: {e}"
