"""
色に関係する画像を作成する。
具体的には、
- 赤、青、白、黒、緑、黄色で埋めた画像
- 赤、青、黒の線を引いた画像
- 赤、青、黒のフォントカラーで、red, blue, blackの文字を書いた画像
- 赤、青、黒のフォントカラーで、赤、青、黒の文字を書いた画像
"""

from PIL import Image, ImageDraw, ImageFont
from itertools import product
from common.path import COLOR_DIR


# 赤、青、黒、白、緑、黄色で埋めた画像
full_colors = [
    ("red", (255, 0, 0)),
    ("blue", (0, 0, 255)),
    ("black", (0, 0, 0)),
    ("white", (255, 255, 255)),
    ("green", (0, 255, 0)),
    ("yellow", (255, 255, 0)),
]

for color_name, color in full_colors:
    img = Image.new("RGB", (100, 100), color)
    img.save(COLOR_DIR / f"filled_{color_name}.png")

# 赤、青、黒の線を引いた画像
colors = full_colors[:3]

for color_name, color in colors:
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line((10, 10, 90, 90), fill=color, width=5)
    img.save(COLOR_DIR / f"line_{color_name}.png")


# 赤、青、黒のフォントカラーで、red, blue, blackの文字を書いた画像
colors = full_colors[:3]
texts = ["red", "blue", "black"]

for text, (color_name, color_code) in product(
    texts, colors
):
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("ヒラギノ明朝 ProN.ttc", 30)
    draw.text((10, 10), text, font=font, fill=color_code, align="center")
    img.save(COLOR_DIR / f"text_color-{color_name}_text-{text}.png")


# 赤、青、黒のフォントカラーで、赤、青、黒の文字を書いた画像
colors = full_colors[:3]
texts = ["赤", "青", "黒"]

for text, (color_name, color_code) in product(
    texts, colors
):
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("ヒラギノ明朝 ProN.ttc", 70)
    draw.text((10, 10), text, font=font, fill=color_code, align="center")
    img.save(COLOR_DIR / f"text_color-{color_name}_text-{text}.png")
