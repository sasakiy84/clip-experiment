

from common.clip import CLIP, Embeddings, Similarity
from common.path import COLOR_DIR, save_similarity_result

from itertools import product, combinations

# 対象のテキストと画像を用意

full_colors = [
    "red",
    "blue",
    "black",
]

# 色の画像
image_paths = [
    COLOR_DIR / f"line_{color}.png" for color in full_colors
]

text_japanese = [
    "赤",
    "青",
    "黒",
]

print("base model")

base_model = CLIP()

text_embeddings: list[Embeddings] = []
for color in full_colors:
    text_embedding = base_model.get_text_embeddings(f"line {color}")
    text_embeddings.append(Embeddings(name=f"a {color} line", type="text", embedding=text_embedding))

for color in text_japanese:
    text_embedding = base_model.get_text_embeddings(f"{color} の線")
    text_embeddings.append(Embeddings(name=f"{color} の線", type="text", embedding=text_embedding))

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

# a の画像から b のテキストの特徴量を引いて c のテキストの特徴量を足すと d の画像の特徴量に近づくかどうかを確認
for a_img, b_text, c_text, d_image in product(image_embeddings, text_embeddings, text_embeddings, image_embeddings):
    if b_text.name == c_text.name:
        continue

    result_embedding = a_img.embedding - b_text.embedding + c_text.embedding
    similarity = CLIP.similarity(result_embedding, d_image.embedding)

    ressult_embedding_name = f"img: {a_img.name} - text: {b_text.name} + text: {c_text.name}"
    similarities.append(
        Similarity(
            a=Embeddings(name=ressult_embedding_name, type="custom", embedding=result_embedding), 
            b=d_image,
            similarity=similarity
        )
    )

for similarity in similarities:
    print(f"{similarity.a.name} and {similarity.b.name} similarity: {similarity.similarity}")

save_similarity_result(similarities, "color-line", "base")


print("")
print("-" * 50)
print("")

print("large model")

large_model = CLIP(
    model_name="openai/clip-vit-large-patch14"
)

text_embeddings: list[Embeddings] = []
for color in full_colors:
    text_embedding = large_model.get_text_embeddings(f"a {color} line")
    text_embeddings.append(Embeddings(name=f"line {color}", type="text", embedding=text_embedding))

for color in text_japanese:
    text_embedding = large_model.get_text_embeddings(f"{color} の線")
    text_embeddings.append(Embeddings(name=f"{color} の線", type="text", embedding=text_embedding))

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

# a の画像から b のテキストの特徴量を引いて c のテキストの特徴量を足すと d の画像の特徴量に近づくかどうかを確認
for a_img, b_text, c_text, d_image in product(image_embeddings, text_embeddings, text_embeddings, image_embeddings):
    if a_img.name == d_image.name:
        continue
    if b_text.name == c_text.name:
        continue

    result_embedding = a_img.embedding - b_text.embedding + c_text.embedding
    similarity = CLIP.similarity(result_embedding, d_image.embedding)

    ressult_embedding_name = f"img: {a_img.name} - text: {b_text.name} + text: {c_text.name}"
    similarities.append(
        Similarity(
            a=Embeddings(name=ressult_embedding_name, type="custom", embedding=result_embedding), 
            b=d_image,
            similarity=similarity
        )
    )



for similarity in similarities:
    print(f"{similarity.a.name} and {similarity.b.name} similarity: {similarity.similarity}")

save_similarity_result(similarities, "color-line", "large")

