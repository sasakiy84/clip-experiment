"""
CLIP のモデルを使って得られるテキストと画像のベクトルの次元を確認する
"""

from common.clip import CLIP
from common.path import IMG_DIR

dog_text = "a dog in a park"
dot_image_path = IMG_DIR / "dog-in-a-park.jpg"


print(
    "base model"
)
base_model = CLIP(
    model_name="openai/clip-vit-base-patch32"
)
dog_text_embedding = base_model.get_text_embeddings(dog_text)
dog_image_embedding = base_model.get_image_embeddings(dot_image_path)
print(
    f'dim of dog_text_embedding: {dog_text_embedding.shape}'
)
print(
    f'dim of dog_image_embedding: {dog_image_embedding.shape}'
)
print("example of dog_text_embedding:")
print(dog_text_embedding[0, :10])
print("example of dog_image_embedding:")
print(dog_image_embedding[0, :10])

print("")
print("-" * 50)
print()


print("large model")
large_model = CLIP(
    model_name="openai/clip-vit-large-patch14"
)
dog_text_embedding = large_model.get_text_embeddings(dog_text)
dog_image_embedding = large_model.get_image_embeddings(dot_image_path)
print(
    f'dim of dog_text_embedding: {dog_text_embedding.shape}'
)
print(
    f'dim of dog_image_embedding: {dog_image_embedding.shape}'
)

print("example of dog_text_embedding:")
print(dog_text_embedding[0, :10])
print("example of dog_image_embedding:")
print(dog_image_embedding[0, :10])


# base model
# dim of dog_text_embedding: torch.Size([1, 512])
# dim of dog_image_embedding: torch.Size([1, 512])
# example of dog_text_embedding:
# tensor([-0.1417,  0.2545, -0.1575,  0.0585,  0.0324,  0.1362,  0.0056, -0.4545,
#         -0.1632,  0.1335], grad_fn=<SliceBackward0>)
# example of dog_image_embedding:
# tensor([-0.1929,  0.1272,  0.2916,  0.1402, -0.0016, -0.3097,  0.5896, -0.0978,
#         -0.0688, -0.0150], grad_fn=<SliceBackward0>)

# --------------------------------------------------

# large model
# dim of dog_text_embedding: torch.Size([1, 768])
# dim of dog_image_embedding: torch.Size([1, 768])
# example of dog_text_embedding:
# tensor([-0.0889, -0.0161,  0.1273, -0.3661,  0.0228,  0.0948,  0.4601,  0.0444,
#          0.0139, -1.2418], grad_fn=<SliceBackward0>)
# example of dog_image_embedding:
# tensor([-0.5174,  0.8016, -0.7575,  0.2863,  0.5159, -0.4443, -0.2799,  0.5796,
#         -0.1195, -0.2888], grad_fn=<SliceBackward0>)
