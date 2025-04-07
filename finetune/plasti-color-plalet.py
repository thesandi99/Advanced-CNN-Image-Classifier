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

class AIVSHumanDataset(Dataset):
    def __init__(self, csv_file, root_dir='', transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            root_dir (str): Directory with all the images. If csv has full paths, root_dir can be ''.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['file_name'])
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, label
    
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # Mean for ImageNet
                         [0.229, 0.224, 0.225])   # Std for ImageNet
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])




# Adjust the path to your CSV file
csv_path = '/kaggle/input/ai-vs-human-generated-dataset/train.csv'

df = pd.read_csv(csv_path)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)



train_dataset = AIVSHumanDataset(
    csv_file='train_split.csv',
    root_dir='/kaggle/input/ai-vs-human-generated-dataset/',  # or your image root if needed
    transform=train_transforms
)

val_dataset = AIVSHumanDataset(
    csv_file='val_split.csv',
    root_dir='/kaggle/input/ai-vs-human-generated-dataset/',  # or your image root if needed
    transform=val_transforms
)

batch_size = 96

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained ResNeXt model
model = models.resnext50_32x4d(pretrained=True)  # or resnext101_32x8d

# Freeze early layers if you want (optional)
for param in model.parameters():
    param.requires_grad = True  # By default, we unfreeze everything for fine-tuning

# Replace the final classification layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes

model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


num_epochs = 5  # adjust as needed

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, device)
    val_loss, val_acc = validate_one_epoch(model, criterion, val_loader, device)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# -----------------------------
# 1. Load test.csv
# -----------------------------
test_df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/test.csv')  


# -----------------------------
# 2. Define a test Dataset
# -----------------------------
class AIVSHumanTestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the file_name from the row
        file_name = '/kaggle/input/ai-vs-human-generated-dataset/' + self.df.iloc[idx]['id']
        # Load image
        image = Image.open(file_name).convert('RGB')
        # Apply transforms (if any)
        if self.transform:
            image = self.transform(image)
        return image

# -----------------------------
# 3. Create test transforms
# -----------------------------
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 4. Instantiate the Dataset & DataLoader
# -----------------------------
test_dataset = AIVSHumanTestDataset(test_df, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------------
# 5. Load your trained model
# -----------------------------
# Example: 
#   model = YourResNeXtModel(...)
#   model.load_state_dict(torch.load('path/to/model.pth'))
#   model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# -----------------------------
# 6. Generate predictions
# -----------------------------
all_preds = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)  # shape [batch_size, 2] if using nn.CrossEntropyLoss
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy().tolist())

# -----------------------------
# 7. Create submission DataFrame
# -----------------------------
# "id" must match the id in test.csv
# "label" is your predicted class (0 or 1)
submission_df = pd.DataFrame({
    'id': test_df['id'],      # from test.csv
    'label': all_preds        # from the model predictions
})

# -----------------------------
# 8. Save submission (no index)
# -----------------------------
submission_df.to_csv('submission.csv', index=False)

print("Submission file saved as submission.csv!")