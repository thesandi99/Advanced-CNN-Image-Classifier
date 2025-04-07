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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from tqdm import tqdm
import matplotlib.pyplot as plt
import timm

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Config:
    seed = 42
    batch_size = 32
    epochs = 1
    base_lr = 0.03
    weight_decay = 1e-9
    momentum = 0.9
    nesterov = True
    image_size = 384
    num_workers = 8

def setup_ddp():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    destroy_process_group()

class TestImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)

class ImageDataset(Dataset):
    def __init__(self, file_list, labels=None, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Calculate stats before transformation
        img_tensor = transforms.ToTensor()(img)  # Convert to tensor for stats
        mean = img_tensor.mean(dim=[1, 2])  # Mean across H,W dimensions
        std = img_tensor.std(dim=[1, 2])    # STD across H,W dimensions
        
        # Print stats for this image
        print(f"Image: {os.path.basename(img_path)}")
        print(f'Mean: {mean.tolist()}')
        print(f'STD: {std.tolist()}')
        
        if self.transform:
            img = self.transform(img)
            
        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        
        return img

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(Config.image_size, interpolation=3),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(Config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

class MyCustomModel(nn.Module):
    def __init__(self, base_model, mean_dim, std_dim):
        super(MyCustomModel, self).__init__()
        self.base_model = base_model  
        # Ek extra layer jo state information process kare
        self.state_layer = nn.Sequential(
            nn.Linear(mean_dim, std_dim , 128),
            nn.ReLU()
        )

        num_ftrs = base_model.fc.in_features

        base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),         
            nn.ReLU(),                     
            nn.Dropout(0.4),               
            nn.Linear(512, 2) 
        )

    
    def forward(self, x,  [mean_dim, std_dim = state]):
        # Image se features nikaalte hain
        features = self.base_model.forward_features(x)
        # Agar base model direct features extract nahi karta, toh aapko customize karna padega
        state_features = self.state_layer(state)
        # Dono ko concatenate karein
        combined = torch.cat([features, state_features], dim=1)
        output = self.classifier(combined)
        return output


def prepare_model():
    # model = models.vit_h_14(weights='IMAGENET1K_SWAG_LINEAR_V1')
    
    # Modify head for binary classification 
    # num_ftrs = model.heads.head.in_features
    # model.heads.head = nn.Sequential(
    #     nn.Linear(num_ftrs, 512),
    #     nn.ReLU(),
    #     nn.Dropout(0.4),
    #     nn.Linear(512, 2)
    # )
    
    # Load pre-trained Regnet
    model = models.resnext101_32x8d(weights='IMAGENET1K_V2')
    model = MyCustomModel(model)
    
    return model

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Initialize optimizer with SGD
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov
        )
        
        # Cosine learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_acc = 0
        
        if self.gpu_id == 0:
            progress = tqdm(self.train_loader, desc="Training")
        else:
            progress = self.train_loader
            
        for data, labels in progress:
            data = data.to(self.gpu_id)
            labels = labels.to(self.gpu_id)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            total_acc += acc
            
        return total_loss / len(self.train_loader), total_acc / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        all_preds = []
        all_labels = []
        
        for data, labels in self.val_loader:
            data = data.to(self.gpu_id)
            labels = labels.to(self.gpu_id)
            
            with autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            total_acc += acc
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        f1 = f1_score(all_labels, all_preds)
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
                preds = outputs.argmax(dim=1).cpu().numpy()  # Move predictions to CPU

            local_predictions.extend(preds)
            local_image_names.extend([f"test_data_v2/{name}" for name in names])  # No GPU-dependent modification

        # **Step 2: Gather results from all processes**
        all_predictions = [None] * dist.get_world_size()
        all_image_names = [None] * dist.get_world_size()
        dist.all_gather_object(all_predictions, local_predictions)
        dist.all_gather_object(all_image_names, local_image_names)

        # **Step 3: Merge predictions only on rank 0**
        if self.gpu_id == 0:
            final_predictions = [pred for sublist in all_predictions for pred in sublist]
            final_image_names = [img for sublist in all_image_names for img in sublist]
            
            submission_df = pd.DataFrame({'id': final_image_names, 'label': final_predictions})
            submission_df.to_csv("submission.csv", index=False)
            
            print("\nSubmission file generated: submission.csv")
            print("\nLabel distribution in predictions:")
            print(submission_df["label"].value_counts())

        return None
    


def create_submission(trainer, test_dataset, config):
    """
    Creates submission file using distributed inference with proper error handling
    """
    try:
        # Create distributed sampler for test data
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            sampler=test_sampler,
            num_workers=config.num_workers,
        )
        print(f"test_loader Dataset size: {len(test_loader)}")
        trainer.predict_test(test_loader)
        
        return None
        
    except Exception as e:
        print(f"Error in create_submission on rank {trainer.gpu_id}: {str(e)}")
        raise e

def main():
    config = Config()
    seed_everything(config.seed)
    setup_ddp()
    
    # Data preparation
    base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
    df_train = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    df_train['file_name'] = df_train['file_name'].apply(lambda x: os.path.join(base_dir, x))

  
    # Load and prepare test data
    df_test = pd.read_csv(os.path.join(base_dir, 'test.csv'))
    df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        df_train['file_name'].values,
        df_train['label'].values,
        test_size=0.1,
        random_state=config.seed,
        shuffle=True
    )
    
    train_transforms, val_transforms = get_transforms()
    
    train_dataset = ImageDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = ImageDataset(val_paths, val_labels, transform=val_transforms)
    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=False
    )
    test_dataset = TestImageDataset(df_test['id'].values, transform=val_transforms)
    model = prepare_model()
    trainer = Trainer(model, train_loader, val_loader, config)
    
    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc, val_f1 = trainer.validate()
        trainer.scheduler.step()
        
        if int(os.environ["LOCAL_RANK"]) == 0:
            print(f"Epoch {epoch+1}/{config.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
       
    print(f"Test Dataset size: {len(test_dataset)}")
    # Create submission after training
    submission_df = create_submission(trainer, test_dataset, config)

    cleanup_ddp()

if __name__ == "__main__":
    main()

