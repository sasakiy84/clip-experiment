from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Literal
from dataclasses import dataclass

MODEL_NAMES = Literal[
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14"
]

@dataclass
class Embeddings:
    name: str
    type: Literal["text", "image", "custom"]
    embedding: torch.Tensor

@dataclass
class Similarity:
    a: Embeddings
    b: Embeddings
    similarity: float

class CLIP:
    def __init__(
        self,
        model_name: MODEL_NAMES = "openai/clip-vit-base-patch32"
    ):
        self.base_model: CLIPModel = CLIPModel.from_pretrained(model_name)
        self.base_processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name)

    def get_text_embeddings(self, text):
        inputs = self.base_processor(text=text, return_tensors="pt", padding=True)
        outputs = self.base_model.get_text_features(**inputs)
        text_embedding = outputs
        return text_embedding

    def get_image_embeddings(self, image_path):
        image = self.preprocess_image(image_path)
        inputs = self.base_processor(images=image, return_tensors="pt", padding=True)
        outputs = self.base_model.get_image_features(**inputs)
        image_embedding = outputs
        return image_embedding
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return image
    
    @staticmethod
    def similarity(embedding1, embedding2):
        return torch.cosine_similarity(embedding1, embedding2).item()
