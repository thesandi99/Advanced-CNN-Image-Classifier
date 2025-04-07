# %%writefile train.py
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import models
from torchvision.transforms import v2 as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Set to False for reproducibility

class Config:
    seed = 42
    batch_size = 64  # Increased batch size
    epochs = 30  # More epochs for better convergence
    base_lr = 1e-3  # Adjusted learning rate
    weight_decay = 5e-5  # Adjusted weight decay
    image_size = 384  # Larger input size for better feature extraction
    num_workers = 8  # Increased workers
    num_classes = 2
    patience = 5  # For early stopping

def setup_ddp():
    init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup_ddp():
    destroy_process_group()

def preprocess_image(img_path, img_size):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        return img
    except Exception as e:
        print(f"Error loading {img_path}: {str(e)}")
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = preprocess_image(self.paths[idx], Config.image_size)
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(image=img)["image"]
            
        return img, torch.tensor(label, dtype=torch.long)

class TestImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = preprocess_image(img_path, Config.image_size)
        
        if self.transform:
            img = self.transform(image=img)["image"]
            
        return img, os.path.basename(img_path)

def get_transforms():
    train_transform = A.Compose([
        A.Resize(Config.image_size, Config.image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussNoise(p=0.3),
        A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(Config.image_size, Config.image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, val_transform

class ConvNextModel(nn.Module):
    def __init__(self, model_name='convnext_large', pretrained=True, num_classes=Config.num_classes):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.dropout = nn.Dropout(0.5)
        # Get the feature dimension from the backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 3, Config.image_size, Config.image_size)
            features = self.backbone(dummy)
            n_features = features.shape[1]
        self.fc = nn.Linear(n_features, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        return self.fc(features)

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.gpu_id = dist.get_rank()
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.base_lr,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-6
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
        self.scaler = GradScaler()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_acc = 0
        
        progress = tqdm(self.train_loader, desc="Training") if self.gpu_id == 0 else self.train_loader
            
        for data, labels in progress:
            data, labels = data.to(self.gpu_id), labels.to(self.gpu_id)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_acc += (preds == labels).float().mean().item()
            
        return total_loss / len(self.train_loader), total_acc / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        all_preds, all_labels = [], []
        
        for data, labels in self.val_loader:
            data, labels = data.to(self.gpu_id), labels.to(self.gpu_id)
            
            with autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_acc += (preds == labels).float().mean().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        f1 = f1_score(all_labels, all_preds, average='macro')
        return total_loss / len(self.val_loader), total_acc / len(self.val_loader), f1
    
    @torch.no_grad()
    def predict_test(self, test_loader):
        self.model.eval()
        local_predictions = []
        local_image_names = []
        
        for data, names in tqdm(test_loader, desc="Predicting", disable=self.gpu_id != 0):
            data = data.to(self.gpu_id)
            with autocast():
                outputs = self.model(data)
            preds = outputs.argmax(dim=1).cpu().numpy()
            local_predictions.extend(preds)
            local_image_names.extend([f"test_data_v2/{name}" for name in names])

        all_predictions = [None] * dist.get_world_size()
        all_image_names = [None] * dist.get_world_size()
        dist.all_gather_object(all_predictions, local_predictions)
        dist.all_gather_object(all_image_names, local_image_names)

        if self.gpu_id == 0:
            final_predictions = [pred for sublist in all_predictions for pred in sublist]
            final_image_names = [img for sublist in all_image_names for img in sublist]
            
            submission_df = pd.DataFrame({'id': final_image_names, 'label': final_predictions})
            submission_df.to_csv("submission.csv", index=False)
            print("\nSubmission file generated: submission.csv")
            print(submission_df["label"].value_counts())

def create_submission(trainer, test_dataset, config):
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    trainer.predict_test(test_loader)

def main():
    config = Config()
    seed_everything(config.seed)
    setup_ddp()
    
    base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
    df_train = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    df_train.drop(columns=["Unnamed: 0"], inplace=True)
    df_train['file_name'] = df_train['file_name'].apply(lambda x: os.path.join(base_dir, x))
    # Use full dataset instead of 10%
    
    df_test = pd.read_csv(os.path.join(base_dir, 'test.csv'))
    df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        df_train['file_name'].values,
        df_train['label'].values,
        test_size=0.2,  # Adjusted split
        random_state=config.seed,
        stratify=df_train['label'].values  # Add stratification
    )
    
    train_transforms, val_transforms = get_transforms()
    
    train_dataset = ImageDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = ImageDataset(val_paths, val_labels, transform=val_transforms)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    model = ConvNextModel(model_name='convnext_large', pretrained=True)
    trainer = Trainer(model, train_loader, val_loader, config)
    
    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc, f1 = trainer.validate()
        trainer.scheduler.step()
        
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}/{config.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                torch.save({
                    'model': trainer.model.module.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                    'epoch': epoch,
                    'scheduler': trainer.scheduler.state_dict()
                }, 'best_model.pth')
                print(f"Saved best model with F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print("Early stopping triggered")
                    break
    
    test_dataset = TestImageDataset(df_test['id'].values, transform=val_transforms)
    create_submission(trainer, test_dataset, config)
    
    cleanup_ddp()

if __name__ == "__main__":
    main()