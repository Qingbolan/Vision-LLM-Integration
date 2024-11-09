import os
from PIL import Image
from medmnist import BreastMNIST
from torchvision.transforms import Resize, ToTensor, Compose

# Define the dataset save path
data_dir = './data/BreastMNIST'
os.makedirs(data_dir, exist_ok=True)

# Define data transformations, including resizing to 227x227
transform = Compose([
    Resize((227, 227)),
    ToTensor()
])

# Download and load the training set
train_dataset = BreastMNIST(split='train', transform=transform, download=True)

# Create positive and negative directories
positive_dir = os.path.join(data_dir, 'positive')
negative_dir = os.path.join(data_dir, 'negative')
raw_dir = os.path.join(data_dir, 'raw')
os.makedirs(positive_dir, exist_ok=True)
os.makedirs(negative_dir, exist_ok=True)

# Save images to the corresponding directories
for idx, (img, label) in enumerate(train_dataset):
    # Move the channel dimension to the last position and convert to NumPy
    img = img.permute(1, 2, 0).numpy() * 255  # Shape: (H, W, C)

    # If the image has a single channel, remove the last dimension
    if img.shape[2] == 1:
        img = img.squeeze(2)  # Shape: (H, W)

    # Convert to unsigned 8-bit integer type
    img = img.astype('uint8')

    # Create a PIL image. Use 'L' mode for grayscale images.
    img_pil = Image.fromarray(img, mode='L')

    # Define the save path based on the label
    if label == 1:
        save_path = os.path.join(positive_dir, f'img_{idx}.png')
    else:
        save_path = os.path.join(negative_dir, f'img_{idx}.png')

    # Save the image
    img_pil.save(save_path)
    # img_pil.save(os.path.join(raw_dir, f'img_{idx}.png'))

    # Optional: Print progress every 100 images
    if (idx + 1) % 100 == 0:
        print(f'Saved {idx + 1} images')

print("All images have been saved successfully.")
