%%writefile train_one_epoch.py
import os
import math
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score,
    accuracy_score
)
import numpy as np
import pandas as pd
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    update_freq = args.update_freq
    use_amp = True  # Assuming use_amp is True
    optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, original_pil_images) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            utils.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        original_pil_images = [img for img in original_pil_images]

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(samples, original_pil_image=original_pil_images) # Pass PIL images
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        loss /= update_freq
        loss_scaler.scale(loss).backward() # Moved backward() inside the loop, before optimizer.step()

        if (data_iter_step + 1) % update_freq == 0:
             if max_norm > 0:  # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
             loss_scaler.step(optimizer)
             loss_scaler.update()
             optimizer.zero_grad()
             if model_ema is not None:
                model_ema.update(model)

        torch.cuda.synchronize()

        # Calculate accuracy (handle Mixup properly)
        if mixup_fn is None:
            class_acc = (output.argmax(dim=-1) == targets).float().mean()
        else:
            # With Mixup, accuracy needs to be calculated differently.
            # Assuming targets are one-hot encoded after mixup.
            _, predicted = output.topk(1, 1, True, True)
            class_acc = predicted.eq(targets.argmax(dim=-1, keepdim=True)).float().mean()


        min_lr = 1e-4
        max_lr = 10e-5
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]


        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        if weight_decay_value is not None:
            metric_logger.update(weight_decay=weight_decay_value)
       

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            # log_writer.update(grad_norm=grad_norm, head="opt") # Also removed here
            log_writer.set_step()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_submission(data_loader, model, device, use_amp=False):
    """Evaluate the model on the submission data loader and create a submission file."""
    model.eval()
    predictions = []
    image_names = []
    confidences = []

    with torch.no_grad():
        for data, labels, names, original_pil_images in tqdm(data_loader, desc="Predicting on Test Data", miniters=10):
            data = data.to(device)
            # We don't need labels, but we have to unpack it. We can ignore it.
            original_pil_images = [img for img in original_pil_images] # Convert to list

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(data, original_pil_image=original_pil_images) # Pass original_pil_images

            # Get predictions (class index)
            preds = output.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())

            # Get confidence scores (probability of class 1)
            confidence = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            confidences.extend(confidence)

            image_names.extend([f"test_data_v2/{name}" for name in names])  # IMPORTANT: Adjust path as needed

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': image_names,
        'label': predictions,
        'confidence': confidences  # Include confidence
    })
    # Create submission DataFrame
    submission_df_prob = pd.DataFrame({
        'id': image_names,
        'label': predictions,
        'confidence': confidences  # Include confidence
    })

    submission_df.to_csv("submission.csv", index=False)
    submission_df_prob.to_csv("submission_prob.csv", index=False)
    print("Submission file 'submission.csv' created.")

    return pd.read_csv('submission.csv')


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    predictions = []
    labels = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]  # Use the correct target index
        original_pil_images = batch[3]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        original_pil_images = [img for img in original_pil_images]


        # compute output
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(images, original_pil_image=original_pil_images) # Pass PIL images
            loss = criterion(output, target)

        predictions.append(output.detach())
        labels.append(target.detach())

        acc1 = accuracy(output, target, topk=(1,))[0]  # Corrected accuracy call

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))


    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)

    if utils.is_dist_avail_and_initialized():
        output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
        dist.all_gather(output_ddp, predictions)
        predictions = torch.cat(output_ddp, dim=0)

        labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
        dist.all_gather(labels_ddp, labels)
        labels = torch.cat(labels_ddp, dim=0)


    y_pred = torch.softmax(predictions, dim=1)[:, 1].cpu().numpy()
    y_true = labels.cpu().numpy()

    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, ap

