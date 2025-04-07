import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import timm  # Using timm for maxvit_t
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.models as models


# Define paths to dataset files
data_path = '/kaggle/input/ai-vs-human-generated-dataset'
train_csv = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'
test_csv = '/kaggle/input/ai-vs-human-generated-dataset/test.csv'

# Load the training and test datasets
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)

print(f'Training dataset shape: {train.shape}')
print(f'Test dataset shape: {test.shape}')

# Preprocess column names for consistency
train = train[['file_name', 'label']]
train.columns = ['id', 'label']
print("Train columns:", train.columns)
print("Test columns:", test.columns)

# Split the training data into training and validation sets (95% train, 5% validation)
train_df, val_df = train_test_split(
    train, 
    test_size=0.05, 
    random_state=42,  
    stratify=train['label'] 
)
print(f'Train shape: {train_df.shape}')
print(f'Validation shape: {val_df.shape}')

# Check class distribution in both sets
print("\nTrain class distribution:")
print(train_df['label'].value_counts(normalize=True))
print("\nValidation class distribution:")
print(val_df['label'].value_counts(normalize=True))


# Define transforms
train_transforms = transforms.Compose([
    transforms.Resize(232),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Dataset classes
class AIImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.loc[idx, 'id'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.dataframe.loc[idx, 'label']
        return image, label

class TestAIImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

# Create datasets
train_dataset = AIImageDataset(train_df, root_dir=data_path, transform=train_transforms)
val_file_list = [os.path.join(data_path, fname) for fname in val_df['id']]
val_labels = val_df['label'].values
val_dataset = TestAIImageDataset(file_list=val_file_list, transform=val_test_transforms)
test_file_list = [os.path.join(data_path, fname) for fname in test['id']]
test_dataset = TestAIImageDataset(file_list=test_file_list, transform=val_test_transforms)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create DataLoaders
num_workers = os.cpu_count()
batch_size = (len(train_dataset) / num_workers ) - (len(test_dataset) - len(val_dataset)) 
batch_size = int(batch_size / 1000)
batch_size = ({len(train_dataset)} / {len(val_dataset)}) - {len(test_dataset)}
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)



# Load pretrained MaxViT Tiny model using timm
model = models.resnext101_32x8d(weights="DEFAULT")

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last two blocks if available (check attribute 'blocks')
if hasattr(model, 'blocks'):
    for block in model.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True




# Get the input feature size of the classifier
if isinstance(model.classifier, nn.Sequential):
    # Find the last Linear layer in Sequential
    for layer in reversed(model.classifier):
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            break
else:
    in_features = model.classifier.in_features  # If it's a single Linear layer

# Replace the classification head
model.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.BatchNorm1d(in_features),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)  # Binary classification
)


# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# Define optimizer, loss, and scheduler
optimizer = torch.optim.AdamW([
    {'params': model.blocks[-2:].parameters(), 'lr': 1e-5} if hasattr(model, 'blocks') else {'params': model.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
])


criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=5, gamma=0.7)

# Training Loop
epochs = 12
train_losses, train_accuracies = [], []
val_losses, val_accuracies, val_f1s = [], [], []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    for data, label in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        preds = output.argmax(dim=1)
        epoch_accuracy += (preds == label).float().mean().item()
    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_pred_classes = []
    val_labels_list = []
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
            data = data.to(device)
            output = model(data)
            batch_labels = val_labels[i * val_loader.batch_size: (i+1)*val_loader.batch_size]
            batch_labels = torch.tensor(batch_labels, device=device)
            loss = criterion(output, batch_labels)
            val_loss += loss.item()
            preds = output.argmax(dim=1)
            val_acc += (preds == batch_labels).float().mean().item()
            val_pred_classes.extend(preds.cpu().numpy())
            val_labels_list.extend(batch_labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_f1 = f1_score(val_labels_list, val_pred_classes, average='binary')
    val_acc_ = accuracy_score(val_labels_list, val_pred_classes)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
 
    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Acc_: {val_acc_:.4f}")
    scheduler.step()


# Test predictions
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

# Generate predictions and logits for the test set
model.eval()
test_logits = []  # To store logits
test_pred_classes = []

with torch.no_grad():
    for data, _ in tqdm(test_loader, desc="Generating Test Predictions"):
        data = data.to(device)
        output = model(data)  # Raw logits (before softmax)
        
        # Save logits
        test_logits.extend(output.cpu().numpy())  # Store raw logits
        
        # Get predicted class (0 or 1)
        preds = output.argmax(dim=1)
        test_pred_classes.extend(preds.cpu().numpy())

# Convert logits to a DataFrame
logits_df = pd.DataFrame(test_logits, columns=['logit_class_0', 'logit_class_1'])
logits_df['id'] = test['id'].values  # Add image IDs for reference

# Save logits to a CSV file
logits_df.to_csv('test_logits.csv', index=False)

# Add predictions to the test DataFrame
test['label'] = test_pred_classes
test[['id', 'label']].to_csv('submission.csv', index=False)

print("Test logits saved to 'test_logits.csv'")
print("Test predictions saved to 'submission.csv'")
print(pd.read_csv('submission.csv')['label'].value_counts())


import matplotlib.pyplot as plt

for i, (data, label) in enumerate(test_loader):
    if i == 880:  # Change this to the desired image index
        img = data[0].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title(f"Label: {'0' if label[0] == 0 else '1'}")
        plt.show()

        # Apply transforms to see the image like the model does
        img_transformed = train_transforms(Image.fromarray((img * 255).astype(np.uint8)))
        img_transformed = img_transformed.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img_transformed)
        plt.title('Transformed Image')
        plt.show()

        # Get the logits for the current image
        output = model(data.to(device))
        logits = output.detach().cpu().numpy()[0]

        # Create a bar plot of the probabilities
        plt.bar([0, 1], [logits[0], logits[1]], color=['red' if label[0] == 0 else 'green', 'green' if label[0] == 1 else 'red'])
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        plt.show()
        break


