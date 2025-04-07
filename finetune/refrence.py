import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms.v2 import GaussianNoise
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch.cuda.amp as amp
from glob import glob
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoImageProcessor, BeitForImageClassification, AutoModelForImageClassification, ResNetForImageClassification, ViTForImageClassification, ConvNextForImageClassification
import shutil

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = csv_file
        self.img_dir     = img_dir #glob(os.path.join(img_dir, "*.jpg"))
        self.transform   = transform
        assert len(self.img_dir) == len(self.img_dir)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image    = Image.open(img_path).convert("RGB")
        label    = torch.tensor(int(self.annotations.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)
            
        return image, label
    

def create_dataloaders(csv_file, img_dir, img_size=(224, 224), batch_size=32, n_fold=0):
    # Define transforms with basic augmentations
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(), 
        transforms.RandomHorizontalFlip()
    ])

    # Initialize dataset
    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    
    # Create train/validation split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    for i, (train_index, val_index) in enumerate(skf.split(np.zeros(len(csv_file)), csv_file.iloc[:, 1].values)):
        if i == n_fold:
            break
            
    train_dataset = Subset(dataset, train_index)
    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, 
                                 transform=transforms.Compose([transforms.Resize(img_size), 
                                                              transforms.ToTensor()]))
    val_dataset = Subset(dataset, val_index)
    print(len(train_dataset))
    print(len(val_dataset))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        
        outputs = model(images).logits[:, :1]
        loss = criterion(outputs, labels)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            outputs = model(images).logits[:, :1]
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    all_outputs = torch.sigmoid(torch.tensor(all_outputs)).numpy()  # Convert logits to probabilities
    
    return epoch_loss, all_labels, all_outputs

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def train_model(csv_file, img_dir, model, model_name, img_size=(224, 224), num_epochs=10, batch_size=32, lr=1e-4, n_fold=0, device='cuda', patience=3, warmup_epochs=0):
    train_loader, val_loader = create_dataloaders(csv_file, img_dir, img_size=img_size, batch_size=batch_size, n_fold=n_fold)
    train_sets = len(train_loader)

    model     = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, train_sets * warmup_epochs, train_sets * num_epochs)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_losses = []
    val_losses   = []
    
    path = model_name + str(n_fold)
    os.makedirs(path, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        val_loss, val_labels, val_outputs = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate metrics on validation set
        val_preds = (val_outputs > 0.5).astype(int)
        f1 = f1_score(val_labels, val_preds)
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1 Score: {f1:.4f}')
        early_stopping(-f1, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        
    # Plot Loss
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()


labels = pd.read_csv ("/kaggle/input/detect-ai-vs-human-generated-images/train.csv")
labels = labels.iloc[:, 1:].copy()
img_dir = "/kaggle/input/ai-vs-human-generated-dataset"
model = ResNetForImageClassification.from_pretrained("google/siglip-base-patch16-224")
batch_size = 64
lr = 1e-4
img_size = (224, 224)
n_fold = 2
train_model(labels, img_dir, model, 'google/siglip-base-patch16-224', img_size=img_size, num_epochs=5, 
            batch_size=batch_size, lr=lr, n_fold=n_fold, patience=3, warmup_epochs=0)
torch.cuda.empty_cache()


def predict(csv_file, img_dir, model, model_name, img_size=(224, 224), batch_size=32, n_fold=0, device='cuda', delete=False):
    model = model.to(device) # load the model into the GPU
    model.load_state_dict(torch.load(os.path.join(model_name + str(n_fold), 'checkpoint.pth')))
    criterion = nn.BCEWithLogitsLoss()
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    test_dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    _, _, outputs = validate(model, test_loader, criterion, device)
    if delete:
        shutil.rmtree(model_name + str(n_fold))
    return outputs

labels = pd.read_csv ("/kaggle/input/detect-ai-vs-human-generated-images/test.csv")
labels['label'] = 1
img_dir = "/kaggle/input/ai-vs-human-generated-dataset"
preds = predict(labels, img_dir, model, 'google/siglip-base-patch16-224', img_size=img_size, batch_size=batch_size, device='cuda', n_fold=n_fold)

labels['label'] = (preds > 0.5).astype(int)
labels.head()