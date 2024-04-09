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

# Specify the number of classes
num_classes = len(marida_train.classes)
print(f"Number of classes: {num_classes}")
# evaluating the model on MARIDA dataset
model.eval()
model.to(device)

# Evaluate the model on the validation set
with torch.no_grad():
    for i in range(len(marida_val)):
        sample = marida_val[i]
        image = sample["image"]
        mask = sample["mask"]

        image = image.to(device).unsqueeze(0)
        mask = mask.to(device).unsqueeze(0)

        output = model(image)
        # Assuming output is the predicted mask
        # You can compare it with the ground truth mask (mask) to evaluate the model's performance
        # You can use metrics like IoU, F1 score, etc. to evaluate the model
        # For example, you can calculate the IoU as follows:
        predicted_mask = output.argmax(dim=1)
        intersection = (predicted_mask & mask).float().sum()
        union = (predicted_mask | mask).float().sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        print(f"IoU: {iou.item()}")
