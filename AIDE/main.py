%%writefile main.py
import os
import argparse
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from timm.optim import create_optimizer  # Changed to timm.optim
from torchvision import transforms  # Using torchvision directly
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from timm.utils import ModelEma
import torch
import random
import numpy as np
from Model import AIDE_Model  # Import the model
from dataset import ImageDataset #, TestImageDataset  # Import datasets. We might not use TestImageDataset during training.
from train_one_epoch import train_one_epoch, evaluate
from timm.data import create_transform
import utils
from torch.cuda.amp import GradScaler, autocast  # Import GradScaler
import open_clip
from tqdm import tqdm
from torchvision.transforms import v2
import kornia.augmentation as K

class Config:  # Use a class for configuration
    def __init__(self):
        self.seed = 42
        self.batch_size = 64  # Per GPU batch size
        self.num_workers = os.cpu_count()
        self.pin_mem = True
        self.output_dir = 'output'
        self.model_ema_decay = 0.9998
        self.model_ema_force_cpu = False
        self.lr = 2e-5 # Initial learning rate
        self.min_lr = 1e-6 # Minimum learning rate
        self.warmup_epochs = 1
        self.epochs = 1
        self.update_freq = 1 # Gradient accumulation steps
        self.clip_grad = 0.1 # Gradient clipping norm
        self.weight_decay = 0.05
        self.smoothing = 0.1
        self.mixup = 0.8
        self.cutmix = 1.0
        self.cutmix_minmax = None
        self.mixup_prob = 1.0
        self.mixup_switch_prob = 0.5
        self.mixup_mode = 'batch'
        self.nb_classes = 2  # Binary classification
        self.save_ckpt_freq = 1 # Save checkpoint every 1 epoch
        self.resume = '' # Path to resume from, e.g., 'output/checkpoint_best.pth'
        self.start_epoch = 0 # Start from this epoch when resuming
        self.eval = False  # Set to True for evaluation only mode
        self.model_ema_eval = False
        self.num_patches = 16 # Number of top-k/bottom-k patches for DCT
        self.use_amp = True # Use automatic mixed precision
        self.data_dir = '/kaggle/input/ai-vs-human-generated-dataset'  # Or your data directory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.momentum = 0.9



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Important for reproducibility

# Custom collate function
def custom_collate(batch):
    # Separate the different elements of the batch
    # Check the length of the first item in the batch.  If it's 3, we're
    # dealing with the test set (no labels). If it's 4, we have labels.
    if len(batch[0]) == 3:  # Test set (no labels)
        images, paths, pil_images = zip(*batch)
        labels = None  # No labels, so set to None
    elif len(batch[0]) == 4: # Train/val set
        images, labels, paths, pil_images = zip(*batch)
        labels = torch.stack(labels, 0) # Stack labels
    else:
        raise ValueError("Unexpected batch structure")

    images = torch.stack(images, 0) # Always stack images

    return images, labels, paths, list(pil_images)

def stats(dataloader):
    conf = Config()
    mean = torch.zeros(3).to(conf.device)
    square = torch.zeros(3).to(conf.device)
    num_pixels = 0
    
    for images, _ in tqdm(dataloader, desc='Mean & STD'):
        images = images.to(conf.device)
        b, c, h, w = images.shape
        num_pixels += b * h * w
        mean += images.sum(dim=[0, 2, 3])  # Sum over batch, height, width
        square += (images ** 2).sum(dim=[0, 2, 3])  # Sum of x**2

    if num_pixels == 0:
        raise ValueError("Error: No pixels found! Check your dataloader.")

    mean /= num_pixels
    variance = (square / num_pixels) - (mean ** 2)
    variance = torch.clamp(variance, min=1e-6)  # Prevent negative values
    std = torch.sqrt(variance)

    mean = mean.detach().cpu()
    std = std.detach().cpu()

    print(f'Mean: {mean.tolist()}')
    print(f'STD: {std.tolist()}')

    return mean.tolist(), std.tolist()

class DivideBy255(object):
    def __call__(self, tensor):
        return tensor / 255.0
    
def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='adamw', help='optimizer')
    parser.add_argument('--local_rank', type=int, default=0)
    parser_args = parser.parse_args()
    args = Config()
    args.opt = parser_args.opt
    args.gpu = parser_args.local_rank
    args.dist_url = "env://"
    init_distributed_mode(args)


    # Setup distributed training
    if args.distributed:
      args.num_tasks = utils.get_world_size()
      args.global_rank = utils.get_rank()
    else:
      args.num_tasks = 1
      args.global_rank = 0

    seed_everything(args.seed + args.global_rank)

    # --- Data Loading ---
    base_dir = args.data_dir
    train_csv_path = os.path.join(base_dir, 'train.csv')
    test_csv_path = os.path.join(base_dir, 'test.csv')
    train_csv_path = train_csv_path.drop(columns=['Unnamed: 0'])

    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    df_test['id'] = df_test['id'].apply(lambda x: os.path.join(base_dir, x))
    df_train['file_name'] = df_train['file_name'].apply(lambda x: os.path.join(base_dir, x))

    all_image_paths = df_train['file_name'].values
    all_labels = df_train['label'].values

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths, all_labels, test_size=0.2, random_state=args.seed, stratify=all_labels
    )
    Perturbations = K.container.ImageSequential(
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
        K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
    )
    # Transformations
    # Transform
    transform = v2.Compose([
        v2.ToImage(),
        DivideBy255(),
        v2.PILToTensor(),
        v2.ConvertImageDtype(torch.float),
    ])
    transforms = torch.nn.Sequential(
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    )
    scripted_transforms = torch.jit.script(transforms)
    # Custom transforms for the DCT part (applied after getting the PIL Image)
    train_transform = transform
    val_transform = transform

    train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ImageDataset(val_paths, val_labels, transform=val_transform)


    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.num_tasks, rank=args.global_rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.num_tasks, rank=args.global_rank, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
        collate_fn=custom_collate  # Use the custom collate function
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
        collate_fn=custom_collate # Use custom collate for val_loader too
    )


    # --- Model, Optimizer, Loss ---
    model = AIDE_Model(num_classes=args.nb_classes)
    model.to(args.device)


    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True) # Add find_unused_parameters
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters}")


    # --- Optimizer and Loss Function ---
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = GradScaler(enabled=args.use_amp) # Use GradScaler for AMP

    if args.mixup > 0.:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # --- Mixup ---
    mixup_fn = None
    if args.mixup > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    model_ema = None
    if args.model_ema_decay > 0:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    # Resume from Checkpoint 
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            loss_scaler.load_state_dict(checkpoint['scaler'])
            if model_ema is not None and 'model_ema' in checkpoint:
                model_ema.ema.load_state_dict(checkpoint['model_ema'])

            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")


    # Training Loop
    if not args.eval:
        print("Starting training...")
        best_acc1 = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, criterion, train_loader, optimizer, args.device, epoch,
                loss_scaler, args.clip_grad, model_ema, mixup_fn, args=args
            )

            # if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            #     utils.save_model(
            #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #         loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema
            #         )

            test_stats, acc, ap = evaluate(val_loader, model_without_ddp, args.device, use_amp=args.use_amp)
            print(f"Accuracy of the network on the {len(val_dataset)} test images: {test_stats['acc1']:.1f}% Acc: {acc}")
            if best_acc1 < test_stats["acc1"]:
                best_acc1 = test_stats["acc1"]
                # utils.save_model(
                #     args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                #     loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {best_acc1:.2f}%')
        print(f"Best accuracy: {best_acc1:.4f}")

    else:
        print("Starting evaluation...")
        test_stats, acc, ap = evaluate(val_loader, model_without_ddp, args.device, use_amp=args.use_amp)
        print(f"Accuracy: {test_stats['acc1']:.4f}, AP: {ap:.4f}")

    print("Starting evaluation...")
    from dataset import TestImageDataset
    test_dataset = TestImageDataset(df_test['id'].values, transform=val_transform)

    # Use SequentialSampler for test data
    test_sampler = SequentialSampler(test_dataset)

    test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=test_sampler,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,
            collate_fn=custom_collate  # Keep custom_collate
    )
    from train_one_epoch import evaluate_submission
    submission_df = evaluate_submission(test_loader, model_without_ddp, args.device, use_amp=args.use_amp)
    print(f"Generated submission file.  Preview:\n{submission_df.head()}")

       # print(f"Accuracy: {test_stats['acc1']:.4f}, AP: {ap:.4f}")

if __name__ == '__main__':
    main()




