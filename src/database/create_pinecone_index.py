import os
import sys
src_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src"))
sys.path.append(src_directory)
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time
from transformers import AutoProcessor, CLIPModel
from data import dataset,request_method

os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

load_dotenv()

def get_index():
    pincone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pincone_api_key)
    INDEX_NAME = "index-search"
    if not pc.has_index(INDEX_NAME):
        new_index = pc.create_index(
            INDEX_NAME, metric="cosine",
            dimension=512,
            spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
            ))
        while True:
            index = pc.describe_index(INDEX_NAME)
            if index.status.get("ready", False):
                new_index = pc.Index(INDEX_NAME)
                return new_index
            else:
                time.sleep(1)
    else:
        new_index = pc.Index(INDEX_NAME)
        return new_index

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
df = dataset.get_df(1800,2000)
for _, dataset in df.iterrows():
    url = dataset['photo_image_url']
    id = dataset['photo_id']
    img = request_method.get_urlimage(url)

    inputs = processor(images=img, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    embeddings = image_features.detach().cpu().numpy().flatten().tolist()
    pincone_index = get_index()
    pincone_index.upsert(
        vectors=[{
            "id":id,
            "values": embeddings,
            "metadata":{
                "url": url,
                "photo_id": id
            }
        }],
        namespace="image-search-dataset",
    )