import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class GIADDataset(Dataset):
    """Custom dataset loader for Ghana Industrial Anomaly Dataset (GIAD)"""
    
    def __init__(self, root_dir, split='train', transform=None, img_size=(256, 256)):
        """
        Args:
            root_dir (string): Directory with all the images
            split (string): 'train' or 'test'
            transform (callable, optional): Optional transforms
            img_size (tuple): Target image dimensions
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.split = split
        
        # Load and organize image paths
        normal_paths = self._get_image_paths(os.path.join(root_dir, 'normal'))
        anomaly_paths = self._get_image_paths(os.path.join(root_dir, 'anomaly'))
        
        # Create labels (0=normal, 1=anomaly)
        normal_labels = [0] * len(normal_paths)
        anomaly_labels = [1] * len(anomaly_paths)
        
        # Combine and split
        all_paths = normal_paths + anomaly_paths
        all_labels = normal_labels + anomaly_labels
        
        # Stratified split (80% train, 20% test)
        train_idx, test_idx = train_test_split(
            range(len(all_paths)),
            test_size=0.2,
            stratify=all_labels,
            random_state=42
        )
        
        if split == 'train':
            self.image_paths = [all_paths[i] for i in train_idx]
            self.labels = [all_labels[i] for i in train_idx]
        else:
            self.image_paths = [all_paths[i] for i in test_idx]
            self.labels = [all_labels[i] for i in test_idx]

    def _get_image_paths(self, folder):
        """Get sorted list of image paths in a folder"""
        return sorted([
            os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default normalization
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
            
        return image, torch.tensor(label, dtype=torch.float32)

def get_data_loaders(config):
    """Create train and test data loaders"""
    transform = get_transforms(config['data']['augmentations'])
    
    train_set = GIADDataset(
        root_dir=config['data']['dataset_path'],
        split='train',
        transform=transform,
        img_size=config['data']['input_size']
    )
    
    test_set = GIADDataset(
        root_dir=config['data']['dataset_path'],
        split='test',
        transform=None,  # No augmentation for test
        img_size=config['data']['input_size']
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['workers']
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['workers']
    )
    
    return train_loader, test_loader