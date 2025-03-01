import requests
from PIL import Image

def get_urlimage(image_url):
    response = requests.get(image_url, stream=True).raw
    image = Image.open(response)
    return image