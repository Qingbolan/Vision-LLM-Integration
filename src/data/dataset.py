import os
from torch.utils.data import Dataset
from PIL import Image

class ConcreteCrackDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(raw_data_path, train_split=0.8):
    negative_dir = os.path.join(raw_data_path, 'Negative')
    positive_dir = os.path.join(raw_data_path, 'Positive')

    negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    file_paths = negative_files + positive_files
    labels = [0]*len(negative_files) + [1]*len(positive_files)

    # Shuffle the dataset
    from sklearn.utils import shuffle
    file_paths, labels = shuffle(file_paths, labels, random_state=42)

    # Split into train and val
    split = int(train_split * len(file_paths))
    train_files, val_files = file_paths[:split], file_paths[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    return train_files, train_labels, val_files, val_labels
