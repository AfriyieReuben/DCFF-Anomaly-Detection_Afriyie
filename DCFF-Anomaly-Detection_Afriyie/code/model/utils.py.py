import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms(aug_config):
    """Create image transformations based on config"""
    transforms = [
        T.ToPILImage(),
        T.Resize(aug_config.get('resize', 256)),
        T.CenterCrop(aug_config.get('crop', 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # Add augmentations if enabled
    if aug_config.get('random_hflip', True):
        transforms.insert(2, T.RandomHorizontalFlip())
    if aug_config.get('random_vflip', False):
        transforms.insert(2, T.RandomVerticalFlip())
    if aug_config.get('color_jitter', False):
        transforms.insert(2, T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ))
    
    return T.Compose(transforms)

def add_gaussian_noise(image, sigma=0.1):
    """Add Gaussian noise to image"""
    noise = torch.randn_like(image) * sigma
    return torch.clamp(image + noise, 0, 1)

def synthetic_defect_overlay(image, defect_mask, prob=0.3):
    """Overlay synthetic defects"""
    if random.random() < prob:
        alpha = random.uniform(0.3, 0.7)
        return image * (1 - alpha) + defect_mask * alpha
    return image

def save_checkpoint(state, filename="checkpoint.pth"):
    """Save model checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    return model, optimizer, epoch