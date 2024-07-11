"""
preprocessor がなにをしているか
"""
from common.path import OBJECT_DIR

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Literal
from dataclasses import dataclass

MODEL_NAMES = Literal[
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14"
]

# 対象のテキストと画像を用意
texts = [
    "TOKYO TOWER",
    "TOKYO SKYTREE",
    "Tour Eiffel",
    "Statue of Liberty",
    "Big Ben"
]

image_paths = [
    OBJECT_DIR / "TOKYO-TOWER.jpg",
    OBJECT_DIR / "TOKYO-SKYTREE.jpg",
    OBJECT_DIR / "Tour-Eiffel.jpg",
    OBJECT_DIR / "Statue-of-Liberty.jpg",
    OBJECT_DIR / "Big-Ben.jpg"
]

# image の前処理を行う

preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open(image_paths[0]).convert("RGB")
inputs = preprocessor(images=image, return_tensors="pt", padding=True)

print(inputs["pixel_values"][0].shape)

# show every pixel value as image
