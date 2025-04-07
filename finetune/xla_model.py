import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import timm
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

# TPU-specific imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.distributed.xla_multiprocessing as xmp

from torch.utils.data import Dataset, DataLoader

class Config:
    BATCH_SIZE = 64  # Increased for TPU efficiency
    SEED = 42
    IMG_SIZE = 224  # Reduced for faster processing
    LR = 3e-4  # Adjusted learning rate for TPU
    NUM_EPOCHS = 15
    WEIGHT_DECAY = 1e-4
    MODEL_PATH = "/kaggle/input/ai-image-classification-lb-0.99939/transformers/default/1/Ai-classification_0.99939.pth"
    DATA_DIR = "/kaggle/input/ai-vs-human-generated-dataset/"
    NUM_CLASSES = 2
    NUM_WORKERS = 8

class CustomDataset(Dataset):
    def __init__(self, df, transforms=None, is_train=True):
        self.df = df
        self.transforms = transforms
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(Config.DATA_DIR, row['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
            
        return image, row['label'] if self.is_train else image

def get_transforms():
    train_transform = albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.3),
        albu.RandomRotate90(p=0.3),
        albu.ShiftScaleRotate(p=0.3),
        albu.RandomBrightnessContrast(p=0.2),
        albu.GaussianBlur(p=0.1),
        albu.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = albu.Compose([
        albu.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_transform

def create_tpu_loader(dataset, batch_size, shuffle=True):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=shuffle
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
        pin_memory=True
    )

def load_model():
    model = timm.create_model('resnext50_32x4d', pretrained=False, num_classes=1)  # Force output to match checkpoint
    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    
    # Modify last layer for 2-class classification
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, Config.NUM_CLASSES)  # Adjust output layer
    
    return model



def train_step(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    para_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    
    for images, labels in para_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        xm.optimizer_step(optimizer)
        xm.mark_step()
        
        total_loss += loss.detach().item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    para_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
    
    with torch.no_grad():
        for images, labels in para_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    # Aggregate across TPU cores
    all_preds = xm.mesh_reduce('preds', torch.cat(all_preds, dim=0), lambda x: x)
    all_labels = xm.mesh_reduce('labels', torch.cat(all_labels, dim=0), lambda x: x)
    
    val_loss = total_loss / len(val_loader)
    val_f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='binary')
    
    return val_loss, val_f1

def train_model(rank, flag):
    # Initialize TPU
    device = xm.xla_device()
    xm.master_print(f'Using device: {device}')
    
    # Data preparation
    train_df = pd.read_csv(os.path.join(Config.DATA_DIR, 'train.csv'))
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.2,
        random_state=Config.SEED, 
        stratify=train_df['label']
    )
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = CustomDataset(train_df, train_transform)
    val_dataset = CustomDataset(val_df, val_transform)
    
    train_loader = create_tpu_loader(train_dataset, Config.BATCH_SIZE)
    val_loader = create_tpu_loader(val_dataset, Config.BATCH_SIZE, shuffle=False)
    
    # Model setup
    model = load_model().to(device)
    optimizer = AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # Handle class imbalance
    class_counts = train_df['label'].value_counts().to_list()
    class_weights = torch.tensor([1/c for c in class_counts], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    best_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    print('START')
    for epoch in range(Config.NUM_EPOCHS):
        # Training
        train_loss = train_step(model, train_loader, optimizer, loss_fn)
        
        # Validation
        val_loss, val_f1 = validate(model, val_loader, loss_fn)
        
        # Save best model
        if xm.is_master_ordinal() and val_f1 > best_f1:
            best_f1 = val_f1
            xm.save(model.state_dict(), 'best_model.pth')
            xm.master_print(f'New best F1: {best_f1:.4f}')
        
        # Logging
        if xm.is_master_ordinal():
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            
            print(f'\nEpoch {epoch+1}/{Config.NUM_EPOCHS}')
            xm.master_print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}')
    
    # Save final model
    if xm.is_master_ordinal():
        torch.save(model.state_dict(), 'final_model.pth')
        visualize_history(history)
    
    return model

def visualize_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Val F1', color='green')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.close()

def predict():
    device = xm.xla_device()
    model = load_model().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))  # Add map_location
    model.eval()

    
    test_df = pd.read_csv(os.path.join(Config.DATA_DIR, 'test.csv'))
    test_dataset = CustomDataset(test_df, get_transforms()[1], is_train=False)
    test_loader = create_tpu_loader(test_dataset, Config.BATCH_SIZE*2, shuffle=False)
    
    all_preds = []
    para_loader = pl.ParallelLoader(test_loader, [device]).per_device_loader(device)
    
    with torch.no_grad():
        for batch in para_loader:
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
    
    # Aggregate predictions from all TPU cores
    all_preds = xm.mesh_reduce('preds', torch.cat(all_preds, dim=0), lambda x: x)
    
    if xm.is_master_ordinal():
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'label': all_preds.cpu().numpy()
        })
        submission_df.to_csv('submission.csv', index=False)
        print('Submission file saved')



device = xm.xla_device()
if __name__ == '__main__':
    # TPU training entry point
    xmp.spawn(train_model, args=(None,), nprocs=1, start_method='fork')
    
    # Run prediction on master core
    if xm.is_master_ordinal():
        predict()