import os
from PIL import Image
from medmnist import ChestMNIST
from torchvision.transforms import Resize, ToTensor, Compose
import numpy as np

# Define the dataset save path
data_dir = './data/ChestMNIST'
os.makedirs(data_dir, exist_ok=True)

# Define data transformations
transform = Compose([
    Resize((227, 227)),
    ToTensor()
])

# Create positive and negative directories
positive_dir = os.path.join(data_dir, 'Positive')
negative_dir = os.path.join(data_dir, 'Negative')
os.makedirs(positive_dir, exist_ok=True)
os.makedirs(negative_dir, exist_ok=True)

# Counters for statistics
image_counter = 0
positive_count = 0
negative_count = 0

# Process all splits and combine them
for split in ['train', 'val', 'test']:
    dataset = ChestMNIST(split=split, transform=transform, download=True)
    print(f"\nProcessing {split} set...")
    print(f"Number of images in {split} set: {len(dataset)}")
    
    # Get raw data to verify labels
    raw_dataset = dataset.imgs
    raw_labels = dataset.labels
    
    print(f"Shape of raw data in {split}: {raw_dataset.shape}")
    print(f"Unique labels in {split}: {np.unique(raw_labels)}")
    
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        
        # The label is already a numpy array, just get the first element
        label_value = label[0]
        
        # Move the channel dimension to the last position and convert to NumPy
        img = img.permute(1, 2, 0).numpy() * 255
        
        # If the image has a single channel, remove the last dimension
        if img.shape[2] == 1:
            img = img.squeeze(2)
        
        # Convert to unsigned 8-bit integer type
        img = img.astype('uint8')
        
        # Create a PIL image
        img_pil = Image.fromarray(img, mode='L')
        
        # Define the save path based on the label
        if label_value == 1:
            save_path = os.path.join(positive_dir, f'img_{image_counter}.png')
            positive_count += 1
        else:
            save_path = os.path.join(negative_dir, f'img_{image_counter}.png')
            negative_count += 1
        
        # Save the image
        img_pil.save(save_path)
        
        image_counter += 1
        
        # Print progress every 100 images
        if image_counter % 100 == 0:
            print(f'Saved {image_counter} images (Positive: {positive_count}, Negative: {negative_count})')

print("\nFinal Statistics:")
print(f"Total images saved: {image_counter}")
print(f"Total positive cases: {positive_count}")
print(f"Total negative cases: {negative_count}")

# Verify files on disk
actual_positive_files = len(os.listdir(positive_dir))
actual_negative_files = len(os.listdir(negative_dir))
print("\nFiles on disk:")
print(f"Positive directory: {actual_positive_files} files")
print(f"Negative directory: {actual_negative_files} files")