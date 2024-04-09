from matplotlib import pyplot as plt

from src.datasets.marida import MARIDA

PATH = "data/MARIDA"

marida = MARIDA(PATH, split="val")
sample = marida[0]

image = sample["image"]
mask = sample["mask"]
print(image.shape, image)
print(mask.shape, mask)

marida.plot(sample)
plt.show()
