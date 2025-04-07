# AI vs Human Image Classifier

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import wandb
import logging
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import logging
from IPython.display import display, HTML

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create console handler that writes to notebook cell output
class IPythonHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        # Use different colors for different log levels
        if record.levelno >= logging.ERROR:
            color = "red"
        elif record.levelno >= logging.WARNING:
            color = "orange"
        else:
            color = "black"
        display(HTML(f'<pre style="color: {color}">{msg}</pre>'))

# Add the custom handler
handler = IPythonHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Now you can use the logger
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")


# Configuration
class Config:
    BASE_PATH = "/path/to/your/dataset"  # Update this with your dataset path
    MODEL_NAME = "resnext50_32x4d"
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 224
    NUM_WORKERS = 4
    USE_WANDB = False

class AIVSHumanDataset(Dataset):
    """Dataset class for AI vs Human generated images classification."""
    
    def __init__(self, csv_file: str, root_dir: str = '', transform: Optional[transforms.Compose] = None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Validate files exist
        self._validate_files()
    
    def _validate_files(self):
        missing_files = []
        for idx, row in self.data.iterrows():
            img_path = os.path.join(self.root_dir, row['file_name'])
            if not os.path.exists(img_path):
                missing_files.append(img_path)
        
        if missing_files:
            logger.error(f"Found {len(missing_files)} missing files")
            raise FileNotFoundError(f"Missing files: {missing_files[:5]}...")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['file_name'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise
            
        label = int(self.data.iloc[idx]['label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class AIVSHumanClassifier:
    """Main classifier class handling training and evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self._setup_model()
        self._setup_training()
        
    def _setup_model(self):
        # Initialize model
        self.model = getattr(models, self.config.MODEL_NAME)(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.config.NUM_CLASSES)
        self.model = self.model.to(self.device)
        
    def _setup_training(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=2, factor=0.1
        )
        
        if self.config.USE_WANDB:
            wandb.init(project="ai-vs-human-detection", name=self.config.MODEL_NAME)
    
    def train_one_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, dataloader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(dataloader.dataset)
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        return val_loss, val_acc, np.array(all_preds), np.array(all_labels)

# Data transforms
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def train_model():
    # Load and split data
    logger.info("Loading and splitting data...")
    df = pd.read_csv(os.path.join(Config.BASE_PATH, 'train.csv'))
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Create datasets
    train_dataset = AIVSHumanDataset(
        train_df, 
        root_dir=Config.BASE_PATH,
        transform=get_transforms(train=True)
    )
    val_dataset = AIVSHumanDataset(
        val_df,
        root_dir=Config.BASE_PATH,
        transform=get_transforms(train=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    # Initialize classifier
    classifier = AIVSHumanClassifier(Config)
    
    # Training loop
    best_val_acc = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(Config.NUM_EPOCHS):
        logger.info(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = classifier.train_one_epoch(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = classifier.validate(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log metrics
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        if Config.USE_WANDB:
            wandb.log(metrics)
            
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.model.state_dict(), 'best_model.pth')
            logger.info(f"Saved new best model with validation accuracy: {val_acc:.4f}")
        
        # Update learning rate
        classifier.scheduler.step(val_loss)
    
    return classifier, train_losses, train_accs, val_losses, val_accs, val_labels, val_preds

def plot_results(train_losses, train_accs, val_losses, val_accs, val_labels, val_preds):
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(val_labels, val_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds))


# Add Test Dataset class
class AIVSHumanTestDataset(Dataset):
    """Dataset class for test predictions."""
    
    def __init__(self, csv_file: str, root_dir: str = '', transform: Optional[transforms.Compose] = None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['id'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise
            
        if self.transform:
            image = self.transform(image)
            
        return image


def predict_test_data():
    """Predict on test data and create submission file."""
    logger.info("Loading test data and model...")
    
    # Load test data
    test_df = pd.read_csv(os.path.join(Config.BASE_PATH, 'test.csv'))
    
    # Create test dataset and dataloader
    test_dataset = AIVSHumanTestDataset(
        test_df,
        root_dir=Config.BASE_PATH,
        transform=get_transforms(train=False)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(models, Config.MODEL_NAME)(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    model.eval()
    
    # Make predictions
    logger.info("Making predictions...")
    all_preds = []
    
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
    
    # Create submission file
    logger.info("Creating submission file...")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': all_preds
    })
    
    # Save submission
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")
    
    return submission_df


if __name__ == "__main__":
    # Train model
    classifier, train_losses, train_accs, val_losses, val_accs, val_labels, val_preds = train_model()
    
    # Plot results
    plot_results(train_losses, train_accs, val_losses, val_accs, val_labels, val_preds)

    # Make predictions on test set
    submission_df = predict_test_data()