from pathlib import Path
import csv
from typing import Literal

from .clip import Similarity

IMG_DIR = Path("img")
IMG_DIR.mkdir(exist_ok=True)

GRAPH_DIR = IMG_DIR / "graph"
GRAPH_DIR.mkdir(exist_ok=True, parents=True)

COLOR_DIR = IMG_DIR / "color"
COLOR_DIR.mkdir(exist_ok=True, parents=True)

OBJECT_DIR = IMG_DIR / "object"
OBJECT_DIR.mkdir(exist_ok=True, parents=True)

RESULT_DIR = Path("result")
RESULT_DIR.mkdir(exist_ok=True)

def save_similarity_result(
        similarities: list[Similarity],
        tag: str,
        model: Literal["base", "large"]
):
    with open(RESULT_DIR / f"sim_{tag}_{model}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["name_a", "type_a", "name_b", "type_b", "similarity", "tag", "model", "a_b_type"])
        for similarity in similarities:
            a_b_type = ""
            if similarity.a.type == "text" and similarity.b.type == "image":
                a_b_type = "text-image"
            elif similarity.a.type == "image" and similarity.b.type == "text":
                a_b_type = "text-image"
            elif similarity.a.type == "text" and similarity.b.type == "text":
                a_b_type = "text-text"
            elif similarity.a.type == "image" and similarity.b.type == "image":
                a_b_type = "image-image"
            elif similarity.a.type == "custom":
                a_b_type = f"custom-{similarity.b.type}"
            elif similarity.b.type == "custom":
                a_b_type = f"custom-{similarity.a.type}"
            else:
                raise ValueError(f"Unknown type: {similarity.a.type}, {similarity.b.type}")

            writer.writerow([
                similarity.a.name,
                similarity.a.type,
                similarity.b.name,
                similarity.b.type,
                similarity.similarity,
                tag,
                model,
                a_b_type
            ])