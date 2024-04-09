import numpy as np
import rasterio
import torch

from src.model import swin_v2

config_path = "checkpoints/swin-v2-msi/config.yaml"
model, transforms, config = swin_v2(config_path)
model.eval()

with rasterio.open("data/test_response.tiff") as src:
    image = src.read()

image_size = (256, 256)

# Crop the image to the size of the model
x = 0
y = 0
width = image_size[0]
height = image_size[1]
image = image[:, y : y + height, x : x + width]

# remove last band
image = image[:12]


image = torch.from_numpy(image.astype(np.float32))
image = image.to("cpu")

transformed_image = transforms(image)
transformed_image = transformed_image.unsqueeze(0)
print(transformed_image.shape)

with torch.no_grad():
    out = model(transformed_image)
    print(f"out: {out}")
    print(out.shape)


out_np = out.cpu().numpy()
print(out_np.shape)

# TODO transform to original shape
