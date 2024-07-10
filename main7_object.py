from common.clip import CLIP, Embeddings, Similarity
from common.path import OBJECT_DIR, save_similarity_result

from itertools import product, combinations

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

coutries = [
    "Japan",
    "France",
    "USA",
    "UK"
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

country_embeddings: list[Embeddings] = []
for country in coutries:
    country_embedding = base_model.get_text_embeddings(country)
    country_embeddings.append(Embeddings(name=country, type="text", embedding=country_embedding))

similarities: list[Similarity] = []
for a, b in product(text_embeddings, image_embeddings):
    similarity = CLIP.similarity(a.embedding, b.embedding)
    similarities.append(Similarity(a=a, b=b, similarity=similarity))

for a, b in product(text_embeddings, country_embeddings):
    similarity = CLIP.similarity(a.embedding, b.embedding)
    similarities.append(Similarity(a=a, b=b, similarity=similarity))

for a, b in product(country_embeddings, image_embeddings):
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


# a の画像から b のテキストの特徴量を引いて c のテキストの特徴量を足すと d の画像の特徴量に近づくかどうかを確認
for a_img, b_text, c_text, d_image in product(image_embeddings, country_embeddings, country_embeddings, image_embeddings):
    if a_img.name == d_image.name:
        continue
    if b_text.name == c_text.name:
        continue

    result_embedding = a_img.embedding - b_text.embedding + c_text.embedding
    similarity = CLIP.similarity(result_embedding, d_image.embedding)
    result_embedding_name = f"img: {a_img.name} - text: {b_text.name} + text: {c_text.name}"

    similarities.append(Similarity(
        a=Embeddings(name=result_embedding_name, type="custom", embedding=result_embedding),
        b=d_image,
        similarity=similarity
    ))

for similarity in similarities:
    print(f"{similarity.a.name} and {similarity.b.name} similarity: {similarity.similarity}")
    
save_similarity_result(similarities, "object", "base")

print("")
print("large")
print("")

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

country_embeddings: list[Embeddings] = []
for country in coutries:
    country_embedding = large_model.get_text_embeddings(country)
    country_embeddings.append(Embeddings(name=country, type="text", embedding=country_embedding))

similarities: list[Similarity] = []

for a, b in product(text_embeddings, image_embeddings):
    similarity = CLIP.similarity(a.embedding, b.embedding)
    similarities.append(Similarity(a=a, b=b, similarity=similarity))

for a, b in product(text_embeddings, country_embeddings):
    similarity = CLIP.similarity(a.embedding, b.embedding)
    similarities.append(Similarity(a=a, b=b, similarity=similarity))

for a, b in product(country_embeddings, image_embeddings):
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

for a_img, b_text, c_text, d_image in product(image_embeddings, country_embeddings, country_embeddings, image_embeddings):
    if a_img.name == d_image.name:
        continue
    if b_text.name == c_text.name:
        continue

    result_embedding = a_img.embedding - b_text.embedding + c_text.embedding
    similarity = CLIP.similarity(result_embedding, d_image.embedding)
    result_embedding_name = f"img: {a_img.name} - text: {b_text.name} + text: {c_text.name}"

    similarities.append(Similarity(
        a=Embeddings(name=result_embedding_name, type="custom", embedding=result_embedding),
        b=d_image,
        similarity=similarity
    ))

for similarity in similarities:
    print(f"{similarity.a.name} and {similarity.b.name} similarity: {similarity.similarity}")

save_similarity_result(similarities, "object", "large")
