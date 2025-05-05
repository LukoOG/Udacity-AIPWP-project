import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import numpy as np
from PIL import Image
train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_Dataloaders(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

    trainloader = DataLoader(train_dataset, batch_size=32)
    validloader = DataLoader(valid_dataset, batch_size=32)
    testloader = DataLoader(test_dataset, batch_size=32)

    return trainloader, validloader, testloader

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image = image.resize((256,256))
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = left + 224
    bottom = top + 224
    
    image = image.crop((left, top, right, bottom))
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (image_array - mean)/ std
    
    return torch.from_numpy(normalized_image.transpose((2,0,1)))