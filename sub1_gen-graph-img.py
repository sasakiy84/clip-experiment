"""
実験に使うグラフの画像を作成する
- y = x
- y = x^2
- y = sin(x)
- y = log(x)
- y = softmax(x)
"""

from matplotlib import pyplot as plt
from common.path import GRAPH_DIR
# グラフの画像を作成

# y = x (range -100 ~ 100)
x = list(range(-100, 101))
y = x
plt.plot(x, y)
plt.grid()
plt.savefig(GRAPH_DIR / "y=x.png")
plt.close()


# y = x^2 (range -100 ~ 100)
x = list(range(-100, 101))
y = [i ** 2 for i in x]
plt.plot(x, y)
plt.grid()
plt.savefig(GRAPH_DIR / "y=x^2.png")
plt.close()

# y = sin(x) (range -10π ~ 10π)
import math
x = [i / 10 for i in range(-10 * 10, 10 * 10)]
y = [math.sin(i) for i in x]
plt.plot(x, y)
plt.grid()
plt.savefig(GRAPH_DIR / "y=sin(x).png")
plt.close()

# y = log(x) (range 0.1 ~ 100)
x = [i / 10 for i in range(1, 1001)]
y = [math.log(i) for i in x]
plt.plot(x, y)
plt.grid()
plt.savefig(GRAPH_DIR / "y=log(x).png")
plt.close()

# y = softmax(x) (range -10 ~ 10)
import torch
x = torch.linspace(-10, 10, 100)
y = torch.nn.functional.softmax(x, dim=0)
plt.plot(x, y)
plt.grid()
plt.savefig(GRAPH_DIR / "y=softmax(x).png")
plt.close()
