

from common.clip import CLIP, Embeddings, Similarity
from common.path import COLOR_DIR, save_similarity_result

from itertools import product, combinations

# 対象のテキストと画像を用意


full_colors = [
    "red",
    "blue",
    "black",
]

text_colors = [
    "赤",
    "青",
    "黒",
]

# 色の画像
image_paths = []
for font_color, text_color in product(full_colors, full_colors + text_colors):
    image_paths.append(COLOR_DIR / f"text_color-{font_color}_text-{text_color}.png")

print("base model")

base_model = CLIP()

text_embeddings: list[Embeddings] = []
for font_color, text_color in product(full_colors, full_colors + text_colors):
    text_embedding = base_model.get_text_embeddings(f"text color {font_color} text {text_color}")
    text_embeddings.append(Embeddings(name=f"text color {font_color} text {text_color}", type="text", embedding=text_embedding))

image_embeddings: list[Embeddings] = []
for image_path in image_paths:
    image_embedding = base_model.get_image_embeddings(image_path)
    image_embeddings.append(Embeddings(name=image_path.name, type="image", embedding=image_embedding))


similarities: list[Similarity] = []
for a, b in product(text_embeddings, image_embeddings):
    similarity = CLIP.similarity(a.embedding, b.embedding)
    similarities.append(Similarity(a=a, b=b, similarity=similarity))

for a, b in combinations(text_embeddings, 2):
    if a == b:
        continue
    similarity = CLIP.similarity(a.embedding, b.embedding)
    similarities.append(Similarity(a=a, b=b, similarity=similarity))

for a, b in combinations(image_embeddings, 2):
    if a == b:
        continue
    similarity = CLIP.similarity(a.embedding, b.embedding)
    similarities.append(Similarity(a=a, b=b, similarity=similarity))

for similarity in similarities:
    print(f"{similarity.a.name} and {similarity.b.name} similarity: {similarity.similarity}")

save_similarity_result(similarities, "color-text", "base")

print("large model")

large_model = CLIP(
    model_name="openai/clip-vit-large-patch14"
)

text_embeddings: list[Embeddings] = []
for font_color, text_color in product(full_colors, full_colors + text_colors):
    text_embedding = large_model.get_text_embeddings(f"text color {font_color} text {text_color}")
    text_embeddings.append(Embeddings(name=f"text color {font_color} text {text_color}", type="text", embedding=text_embedding))

image_embeddings: list[Embeddings] = []
for image_path in image_paths:
    image_embedding = large_model.get_image_embeddings(image_path)
    image_embeddings.append(Embeddings(name=image_path.name, type="image", embedding=image_embedding))

similarities: list[Similarity] = []

for a, b in product(text_embeddings, image_embeddings):
    similarity = CLIP.similarity(a.embedding, b.embedding)
    similarities.append(Similarity(a=a, b=b, similarity=similarity))

for a, b in combinations(text_embeddings, 2):
    if a == b:
        continue
    similarity = CLIP.similarity(a.embedding, b.embedding)
    similarities.append(Similarity(a=a, b=b, similarity=similarity))

for a, b in combinations(image_embeddings, 2):
    if a == b:
        continue
    similarity = CLIP.similarity(a.embedding, b.embedding)
    similarities.append(Similarity(a=a, b=b, similarity=similarity))
    

for similarity in similarities:
    print(f"{similarity.a.name} and {similarity.b.name} similarity: {similarity.similarity}")

save_similarity_result(similarities, "color-text", "large")