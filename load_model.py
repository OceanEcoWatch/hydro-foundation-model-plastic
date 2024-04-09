import torch

from src.model import swin_v2

config_path = "checkpoints/swin-v2-msi/config.yaml"
model, transforms, config = swin_v2(config_path)
model.eval()


# Reset the classifier head to your desired number of classes
model.reset_classifier(num_classes=10)

# Extract image level embeddings
x = torch.randn(1, 12, 256, 256)
x = transforms(x)
model.forward_features(x)  # (1, 1024)

# Extract intermediate feature maps
x = torch.randn(1, 12, 256, 256)
x = transforms(x)
features = model.get_intermediate_layers(x, n=(0, 1, 2, 3, 4), reshape=True)
for i, f in enumerate(features):
    print(i, f.shape)

# with rasterio.open("test_response.tiff") as src:
#     image = src.read()

# image_size = (256, 256)

# # Crop the image to the size of the model
# x = 0
# y = 0
# width = image_size[0]
# height = image_size[1]
# image = image[:, y : y + height, x : x + width]

# # remove last band
# image = image[:12]


# image = torch.from_numpy(image.astype(np.float32))
# image = image.to("cpu")

# transformed_image = transforms(image)
# transformed_image = transformed_image.unsqueeze(0)
# print(transformed_image.shape)

# with torch.no_grad():
#     out = model(transformed_image)
#     print(out.shape)

# # Assuming `out` is your output tensor and `src` is your source raster file
# out_np = out.cpu().numpy()  # Convert the output tensor to a numpy array
# print(out_np.shape)  # Should be (1, 12, 256, 256)
