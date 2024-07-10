"""
graph の embedding を比較する
"""

from common.clip import CLIP, Embeddings, Similarity
from common.path import GRAPH_DIR, save_similarity_result

from itertools import product, combinations

# 対象のテキストと画像を用意
texts = [
    "y = x",
    "y = x^2",
    "y = sin(x)",
    "y = log(x)",
    "y = softmax(x)",
    "graph",
    "グラフ",
    "apple",
    "りんご",
]
image_paths = [
    GRAPH_DIR / "y=x.png",
    GRAPH_DIR / "y=x^2.png",
    GRAPH_DIR / "y=sin(x).png",
    GRAPH_DIR / "y=log(x).png",
    GRAPH_DIR / "y=softmax(x).png"
]

print("base model")

base_model = CLIP()

text_embeddings: list[Embeddings] = []
for text in texts:
    text_embedding = base_model.get_text_embeddings(text)
    text_embeddings.append(Embeddings(name=text, type="text", embedding=text_embedding))

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

save_similarity_result(similarities, "graph", "base")


print("")
print("-" * 50)
print()

print("large model")

large_model = CLIP(
    model_name="openai/clip-vit-large-patch14"
)

text_embeddings: list[Embeddings] = []
for text in texts:
    text_embedding = large_model.get_text_embeddings(text)
    text_embeddings.append(Embeddings(name=text, type="text", embedding=text_embedding))

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

save_similarity_result(similarities, "graph", "large")