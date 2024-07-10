"""
URL の画像を取得する
"""

import requests

from common.path import OBJECT_DIR

# 画像の URL
urls: tuple[str, str] = [
    ("TOKYO TOWER", "https://upload.wikimedia.org/wikipedia/commons/e/ed/TaroTokyo20110213-TokyoTower-01min.jpg"),
    ("TOKYO SKYTREE", "https://upload.wikimedia.org/wikipedia/commons/8/84/Tokyo_Skytree_2014_%E2%85%A2.jpg"),
    ("Tour Eiffel", "https://upload.wikimedia.org/wikipedia/commons/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"),
    ("Statue of Liberty", "https://upload.wikimedia.org/wikipedia/commons/e/e3/USA-NYC-Statue_of_Liberty.jpg"),
    ("Big Ben", "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/Elizabeth_clock%2C_July_2024.jpg/400px-Elizabeth_clock%2C_July_2024.jpg"),
]

# 画像を保存
for name, url in urls:
    # request with wikipedia user-agent
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    with (OBJECT_DIR / f"{"-".join(name.split())}.jpg").open("wb") as f:
        f.write(response.content)

