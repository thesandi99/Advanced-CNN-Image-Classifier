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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Important for reproducibility


class Config:
    seed = 42
    batch_size = 16
    base_lr = 0.0025
    weight_decay = 1e-5
    momentum = 0.9
    nesterov = True
    num_workers = 4
    epochs = 4
    stages = 2
    min_image_size = 128
    max_image_size = 480
    amp = True
    gradient_accumulation_steps = 1
    # GridDistortion is optional. If you use it, uncomment these:
    # min_grid_distort_limit = 0.0
    # max_grid_distort_limit = 0.3

def setup_ddp():
    try:
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    except Exception as e:
        print(f"Error initializing DDP: {e}")
        raise

def cleanup_ddp():
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Error cleaning up DDP: {e}")

class ImageDataset(Dataset):
    def __init__(self, file_list, labels=None, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:  # More robust exception handling
            print(f"Warning: Corrupted/missing image: {img_path}. Using a blank image.")
            img = np.zeros((Config.max_image_size, Config.max_image_size, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']

        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            return img, os.path.basename(img_path)

    def set_transform(self, transform):
        self.transform = transform


import albumentations as A
import random
import numpy as np

class RandAugment(A.ImageOnlyTransform):  # Inherit from ImageOnlyTransform
    def __init__(self, num_ops=2, magnitude=9, always_apply=False, p=0.5):  # p=0.5 default
        super().__init__(always_apply, p)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.augmentations = [
            # Geometric Transforms (with magnitude handling)
            {"transform": A.Rotate, "limit": (-30, 30)},
            {"transform": A.Affine, "shear": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
            {"transform": A.ShiftScaleRotate, "shift_limit": (-0.45, 0.45), "scale_limit": (-0.1, 0.1), "rotate_limit": (-30, 30)},

            # Color/Intensity Transforms (with magnitude handling)
            {"transform": A.ColorJitter, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
            {"transform": A.RandomBrightnessContrast, "brightness_limit": (-0.2, 0.2), "contrast_limit": (-0.2, 0.2)},
            {"transform": A.Solarize, "threshold": (0, 255)}, # Threshold needs special handling
            {"transform": A.Posterize, "num_bits": (4, 8)},   # num_bits needs special handling
            {"transform": A.Equalize},
            {"transform": A.GaussianBlur, "blur_limit":(3,7)},
        ]

        self.interpolation = cv2.INTER_LINEAR
        self.fill_value = 0  # Or another appropriate fill value


    def apply(self, image, **params):
        # Randomly select 'num_ops' augmentations
        ops = random.sample(self.augmentations, self.num_ops)

        for op_dict in ops:
            transform_class = op_dict["transform"]

            # Handle magnitude and apply the transformation
            if transform_class in [A.Rotate, A.Affine, A.TranslateX, A.TranslateY]:
                param_name = list(op_dict.keys())[1]
                min_val, max_val = op_dict[param_name]
                val = min_val + (max_val - min_val) * (self.magnitude / 30.0)  # Scale magnitude
                if transform_class == A.Affine:
                    transform = transform_class(**{param_name: val, "p": 1.0, 'interpolation': self.interpolation, 'border_mode': cv2.BORDER_CONSTANT, 'value':self.fill_value})
                else:
                    transform = transform_class(**{param_name: val, "p": 1.0, 'interpolation': self.interpolation, 'border_mode': cv2.BORDER_CONSTANT, 'value':self.fill_value})

            elif transform_class in [A.ColorJitter, A.RandomBrightnessContrast]:
                #  ColorJitter, BrightnessContrast: Adjust parameters based on magnitude
                transform = transform_class(p=1.0) # Apply with adjusted params in __init__
                for param_name in ["brightness", "contrast", "saturation", "hue","brightness_limit","contrast_limit"]:
                    if param_name in op_dict:
                        if isinstance(op_dict[param_name],tuple): # Handle the limits for brightness/Contrast limit
                            min_val, max_val = op_dict[param_name]
                            val = min_val + (max_val - min_val) * (self.magnitude / 30.0)
                            transform.kwargs[param_name] = (-val, val)

                        else:
                            val = op_dict[param_name] * (self.magnitude / 30.0)  # scale magnitude
                            transform.kwargs[param_name] = val
            elif transform_class == A.Solarize:
                threshold = int(255 * (1 - self.magnitude / 30.0))
                transform = A.Solarize(threshold=threshold, p=1.0)

            elif transform_class == A.Posterize:
                num_bits = int(8 - 4 * (self.magnitude / 30.0))  # Example: scale down bits
                num_bits = max(1, min(num_bits, 8)) # Ensure valid range
                transform = A.Posterize(num_bits=num_bits, p=1.0)
            elif transform_class == A.GaussianBlur:
                transform = transform_class(p=1.0) # Apply the blur transform
                sigma = (self.magnitude / 30.0) * 2.0 + 1e-6 # Scale sigma with magnitude
                transform.kwargs['sigma'] = (sigma, sigma)

            else:  # For transforms like Equalize
                transform = transform_class(p=1.0)

            # Apply the transform using the Albumentations dictionary format
            image = transform(image=image)['image']

        return image

def get_transforms(image_size, randaug_magnitude=None, grid_distort_limit=None, dropout_rate=None, mixup_alpha=None):
    train_transforms_list = [
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
        A.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]

    if randaug_magnitude is not None:
        train_transforms_list.append(RandAugment(num_ops=2, magnitude=randaug_magnitude, p=0.5))  # Use custom RandAugment

    if grid_distort_limit is not None:
        train_transforms_list.append(A.GridDistortion(num_steps=5, distort_limit=grid_distort_limit, interpolation=cv2.INTER_LINEAR, p=0.5))

    train_transforms_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_transforms = A.Compose(train_transforms_list)

    val_transforms = A.Compose([
        A.Resize(Config.max_image_size, Config.max_image_size, interpolation=cv2.INTER_CUBIC),
        A.CenterCrop(Config.max_image_size, Config.max_image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transforms, val_transforms

def prepare_model():
    model = models.efficientnet_v2_l(weights="EfficientNet_V2_L_Weights.IMAGENET1K_V1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[-3:].parameters():
        param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features=1280, out_features=2),
    )
    model.to(device)
    return model

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.epochs_per_stage = config.epochs // config.stages
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=1e-6)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler(enabled=config.amp)
        self.global_step = 0

    def _create_dataloaders(self, image_size, randaug_magnitude, grid_distort_limit, dropout_rate, mixup_alpha):
        train_transforms, val_transforms = get_transforms(image_size, randaug_magnitude, grid_distort_limit, dropout_rate, mixup_alpha)
        self.train_dataset.set_transform(train_transforms)
        self.val_dataset.set_transform(val_transforms)
        train_sampler = DistributedSampler(self.train_dataset, shuffle=True, seed=self.config.seed, drop_last=True)
        val_sampler = DistributedSampler(self.val_dataset, shuffle=False, drop_last=True)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return train_loader, val_loader

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_acc = 0
        if self.gpu_id == 0:
            progress = tqdm(train_loader, desc="Training")
        else:
            progress = train_loader
        for step, (data, labels) in enumerate(progress):
            data = data.to(self.gpu_id)
            labels = labels.to(self.gpu_id)
            with autocast(enabled=self.config.amp):
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss = loss / self.config.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            total_acc += acc
            if self.gpu_id == 0:
                progress.set_postfix({"loss": loss.item() * self.config.gradient_accumulation_steps, "acc": acc})
        return total_loss / len(train_loader), total_acc / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        all_preds = []
        all_labels = []
        for data, labels in val_loader:
            data = data.to(self.gpu_id)
            labels = labels.to(self.gpu_id)
            with autocast(enabled=self.config.amp):
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            total_acc += acc
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        f1 = f1_score(all_labels, all_preds)
        return total_loss / len(val_loader), total_acc / len(val_loader), f1

    @torch.no_grad()
    def predict_test(self, test_loader):
        self.model.eval()
        local_predictions = []
        local_image_names = []
        for data, names in tqdm(test_loader, desc="Predicting", disable=self.gpu_id != 0):
            data = data.to(self.gpu_id)
            with autocast(enabled=self.config.amp):
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
            final_predictions_tensor = torch.tensor(final_predictions, dtype=torch.long, device='cpu')
            submission_df = pd.DataFrame({'id': final_image_names, 'label': final_predictions_tensor.numpy()})
            submission_df.to_csv("submission.csv", index=False)
            print("\nSubmission file generated: submission.csv")
            print("\nLabel distribution in predictions:")
            print(submission_df["label"].value_counts())
        dist.barrier()

    def train(self):
        for stage in range(self.config.stages):
            image_size = int(self.config.min_image_size + (self.config.max_image_size - self.config.min_image_size) * stage / (self.config.stages - 1))
            randaug_magnitude = min(5 + int(10 * stage / (self.config.stages - 1)), 15)
            dropout_rate = 0.2 + 0.2 * stage / (self.config.stages - 1)
            mixup_alpha = 0.0 + 0.2 * stage / (self.config.stages - 1)
            grid_distort_limit = None  # Or calculate based on stage if using
            if hasattr(self.config, 'min_grid_distort_limit'): # Check if using GridDistortion
                grid_distort_limit = self.config.min_grid_distort_limit + (self.config.max_grid_distort_limit - self.config.min_grid_distort_limit) * stage / (self.config.stages - 1)

            train_loader, val_loader = self._create_dataloaders(image_size, randaug_magnitude, grid_distort_limit, dropout_rate, mixup_alpha)

            if stage == 1:
                for param in self.model.module.features[-4].parameters():
                    param.requires_grad = True
            elif stage == 2:
                for param in self.model.module.features[-5].parameters():
                    param.requires_grad = True

            if self.gpu_id == 0:
                print(f"Starting Stage {stage+1}: Image Size = {image_size}, RandAug Magnitude = {randaug_magnitude}, Dropout = {dropout_rate}")

            for epoch in range(self.epochs_per_stage):
                train_loader.sampler.set_epoch(epoch + stage * self.epochs_per_stage)
                train_loss, train_acc = self.train_epoch(train_loader)
                dist.barrier()
                val_loss, val_acc, val_f1 = self.validate(val_loader)
                self.scheduler.step()
                dist.barrier()
                if self.gpu_id == 0:
                    print(f"Epoch {epoch+1 + stage * self.epochs_per_stage}/{self.config.epochs}")
                    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

def main():
    config = Config()
    seed_everything(config.seed)
    setup_ddp()

    base_dir = '/kaggle/input/ai-vs-human-generated-dataset'
    df_train = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    df_train.drop(columns=["Unnamed: 0"], inplace=True)
    df_train['file_name'] = df_train['file_name'].apply(lambda x: os.path.join(base_dir, x))
    df_test = pd.read_csv(os.path.join(base_dir, 'test.csv'))
    df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))
    df_train = df_train.sample(frac=0.01, random_state=config.seed)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        df_train['file_name'].values,
        df_train['label'].values,
        test_size=0.1,
        random_state=config.seed,
        shuffle=True,
        stratify=df_train['label'].values
    )

    train_dataset = ImageDataset(train_paths, train_labels)
    val_dataset = ImageDataset(val_paths, val_labels)
    test_dataset = ImageDataset(df_test['id'].values)

    model = prepare_model()
    trainer = Trainer(model, train_dataset, val_dataset, config)

    _, val_transforms = get_transforms(Config.max_image_size)  # Only need val_transforms
    test_dataset.set_transform(val_transforms)
    # Create test_loader *once* here, outside the training loop.
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    trainer.train()  # Train the model

    # Predict *after* training is complete.
    trainer.predict_test(test_loader)

    cleanup_ddp()

if __name__ == "__main__":
    main()
# %%
