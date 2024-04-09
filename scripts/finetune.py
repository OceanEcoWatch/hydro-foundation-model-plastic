import rasterio
import torch

from src.datasets.marida import MARIDA
from src.model import swin_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


marida_train = MARIDA("data/MARIDA", split="train")
marida_val = MARIDA("data/MARIDA", split="val")
marida_test = MARIDA("data/MARIDA", split="test")

# Load the pre-trained model
config_path = "checkpoints/swin-v2-msi/config.yaml"
model, transforms, config = swin_v2(config_path)

num_classes = len(marida_train.classes)
model.reset_classifier(num_classes=num_classes)


model.to(device)

# Evaluate the model on the validation set
with torch.no_grad():
    for i in range(len(marida_val)):
        with rasterio.open(marida_val.images[i]) as src:
            transform = src.transform
            crs = src.crs

        sample = marida_val[i]
        image = sample["image"]

        mask = sample["mask"]

        extra_channel = torch.zeros_like(image[0]).unsqueeze(0)
        image = torch.cat(
            [image, extra_channel], dim=0
        )  # TODO Check if this is correct
        image = image.to(device).unsqueeze(0)
        mask = mask.to(device).unsqueeze(0)
        print(image.shape, mask.shape)
        output = model(image)

        print(output.shape)
        predicted_classes = output.argmax(dim=1)
        print(predicted_classes.shape)
        print(predicted_classes)

        # compare output and mask
