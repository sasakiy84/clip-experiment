"""
Word2Vecのように、ベクトルの加算と減算ができるかどうかを確認する
"""

from common.clip import CLIP
from common.path import IMG_DIR

# 対象のテキストと画像を用意
dog_in_park_text = "a dog in a park"
dot_image_path = IMG_DIR / "dog-in-a-park.jpg"
cat_in_park_text = "a cat in a park"
cat_image_path = IMG_DIR / "cat-in-a-park.jpg"


print("base model")

base_model = CLIP()

dog_text_embedding = base_model.get_text_embeddings(dog_in_park_text)
dog_image_embedding = base_model.get_image_embeddings(dot_image_path)
cat_text_embedding = base_model.get_text_embeddings(cat_in_park_text)
cat_image_embedding = base_model.get_image_embeddings(cat_image_path)

# ベクトルの類似度を計算
print("dog and dog-img similarity:", CLIP.similarity(dog_text_embedding, dog_image_embedding))
print("dog and cat-img similarity:", CLIP.similarity(dog_text_embedding, cat_image_embedding))
print("cat and cat-img similarity:", CLIP.similarity(cat_text_embedding, cat_image_embedding))
print("cat and dog-img similarity:", CLIP.similarity(cat_text_embedding, dog_image_embedding))

# dog - img - dog + cat = cat-img ?
result_embedding = dog_image_embedding - dog_text_embedding + cat_text_embedding
print("result and cat-img similarity:", CLIP.similarity(result_embedding, cat_image_embedding))


print("")
print("-" * 50)
print()

print("large model")

large_model = CLIP(
    model_name="openai/clip-vit-large-patch14"
)

dog_text_embedding = large_model.get_text_embeddings(dog_in_park_text)
dog_image_embedding = large_model.get_image_embeddings(dot_image_path)
cat_text_embedding = large_model.get_text_embeddings(cat_in_park_text)
cat_image_embedding = large_model.get_image_embeddings(cat_image_path)

# ベクトルの類似度を計算
print("dog and dog-img similarity:", CLIP.similarity(dog_text_embedding, dog_image_embedding))
print("dog and cat-img similarity:", CLIP.similarity(dog_text_embedding, cat_image_embedding))
print("cat and cat-img similarity:", CLIP.similarity(cat_text_embedding, cat_image_embedding))
print("cat and dog-img similarity:", CLIP.similarity(cat_text_embedding, dog_image_embedding))

# dog-img - dog + cat = cat-img ?
result_embedding = dog_image_embedding - dog_text_embedding + cat_text_embedding
print("result and cat-img similarity:", CLIP.similarity(result_embedding, cat_image_embedding))


# base model
# dog and dog-img similarity: 0.27299198508262634
# dog and cat-img similarity: 0.23436613380908966
# cat and cat-img similarity: 0.2757304310798645
# cat and dog-img similarity: 0.21589361131191254
# result and cat-img similarity: 0.779753565788269

# --------------------------------------------------

# large model
# dog and dog-img similarity: 0.22109100222587585
# dog and cat-img similarity: 0.1579592376947403
# cat and cat-img similarity: 0.2090512216091156
# cat and dog-img similarity: 0.15709587931632996
# result and cat-img similarity: 0.656259298324585
