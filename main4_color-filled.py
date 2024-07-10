"""
色に関する実験をする
"""


from common.clip import CLIP, Embeddings, Similarity
from common.path import COLOR_DIR, save_similarity_result

from itertools import product, combinations

# 対象のテキストと画像を用意

full_colors = [
    "red",
    "blue",
    "black",
    "white",
    "green",
    "yellow",
]

# 色の画像
image_paths = [
    COLOR_DIR / f"filled_{color}.png" for color in full_colors
]

text_japanese = [
    "赤",
    "青",
    "黒",
    "白",
    "緑",
    "黄色",
]

print("base model")

base_model = CLIP()

text_embeddings: list[Embeddings] = []
for color in full_colors:
    text_embedding = base_model.get_text_embeddings(f"filled {color}")
    text_embeddings.append(Embeddings(name=f"filled {color}", type="text", embedding=text_embedding))

for color in text_japanese:
    text_embedding = base_model.get_text_embeddings(f"{color} の背景")
    text_embeddings.append(Embeddings(name=f"{color} の背景", type="text", embedding=text_embedding))

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

save_similarity_result(similarities, "color-filled", "base")


print("")
print("-" * 50)
print()

print("large model")

large_model = CLIP(
    model_name="openai/clip-vit-large-patch14"
)

text_embeddings: list[Embeddings] = []
for color in full_colors:
    text_embedding = large_model.get_text_embeddings(f"filled {color}")
    text_embeddings.append(Embeddings(name=f"filled {color}", type="text", embedding=text_embedding))

for color in text_japanese:
    text_embedding = large_model.get_text_embeddings(f"{color} の背景")
    text_embeddings.append(Embeddings(name=f"{color} の背景", type="text", embedding=text_embedding))

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

save_similarity_result(similarities, "color-filled", "large")