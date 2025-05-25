# train.py (in the project root directory)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import time
import os

# Project-specific imports
from src.dataset import PascalVOCDataset, SSDTransform, collate_fn
from src.model import SSDResNet50
from src.loss import MultiBoxLoss
from src.config import (
    DEVICE, NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE, VAL_BATCH_SIZE, # VAL_BATCH_SIZE is used
    LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, NUM_EPOCHS,
    DATA_DIR, MODEL_SAVE_PATH, VOC_CLASSES, IDX_TO_CLASS
)
from src.utils import cxcywh_to_xyxy

# For mAP calculation (using torchmetrics)
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def train_one_epoch(model, data_loader, loss_fn, optimizer, epoch, device, print_freq=10, grad_clip_norm=1.0):
    model.train()
    total_loss_epoch = 0
    total_loc_loss_epoch = 0
    total_conf_loss_epoch = 0
    start_time = time.time()
    
    for i, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        gt_boxes_batch = [target['boxes'].to(device) for target in targets]
        gt_labels_batch = [target['labels'].to(device) for target in targets]

        predicted_locs, predicted_scores = model(images)
        loss, loc_loss, conf_loss = loss_fn(predicted_locs, predicted_scores, gt_boxes_batch, gt_labels_batch)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"!!! NaN or Inf loss detected at Epoch {epoch+1}, Batch {i+1}. Value: {loss.item()}. Skipping batch. !!!")
            del predicted_locs, predicted_scores, loss
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm is not None:
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        
        total_loss_epoch += loss.item()
        total_loc_loss_epoch += loc_loss 
        total_conf_loss_epoch += conf_loss
        
        if (i + 1) % print_freq == 0 or i == 0 or (i + 1) == len(data_loader):
            batch_time_avg = (time.time() - start_time) / (i + 1) if i > 0 else (time.time() - start_time)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{i+1}/{len(data_loader)}] "
                  f"Loss: {loss.item():.4f} (Loc: {loc_loss:.4f} Conf: {conf_loss:.4f}) "
                  f"Batch Time Avg: {batch_time_avg:.3f}s")
    
    num_batches = len(data_loader)
    if num_batches == 0:
        print("Warning: train_loader was empty. No training occurred in this epoch.")
        return 0.0, 0.0, 0.0

    avg_loss = total_loss_epoch / num_batches
    avg_loc_loss = total_loc_loss_epoch / num_batches
    avg_conf_loss = total_conf_loss_epoch / num_batches
    epoch_time = time.time() - start_time
    print(f"--- Epoch {epoch+1} Training Summary ---")
    print(f"Average Loss: {avg_loss:.4f} (Loc: {avg_loc_loss:.4f} Conf: {avg_conf_loss:.4f})")
    print(f"Time taken: {epoch_time:.2f}s")
    return avg_loss, avg_loc_loss, avg_conf_loss


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device, epoch, anchors_cxcywh_for_eval, num_classes_for_eval):
    model.eval()
    total_loss_epoch = 0
    total_loc_loss_epoch = 0
    total_conf_loss_epoch = 0

    # Initialize metric. Torchmetrics will generally try to keep its state on CPU if inputs are CPU.
    metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True)
    # To be absolutely sure metric state is on CPU, you could do:
    # metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to("cpu")
    # However, passing CPU tensors to update() usually suffices.

    print(f"\n--- Epoch {epoch+1} Validation ---")
    start_time = time.time()
    variance = torch.tensor([0.1, 0.1, 0.2, 0.2], device=device) # Keep on GPU for decoding speed

    for i_eval, (images, targets) in enumerate(data_loader):
        images = images.to(device) # Model forward pass on GPU

        # Ground truth (targets) will be moved to CPU later before metric update
        # Store them temporarily for loss calculation on GPU if needed (though loss_fn also moves them)
        gt_boxes_batch_eval_gpu = [target['boxes'].to(device) for target in targets]
        gt_labels_batch_eval_gpu = [target['labels'].to(device) for target in targets]

        predicted_locs, predicted_scores = model(images) # Inference on GPU
        
        # Calculate validation loss (can still be on GPU for speed if loss_fn handles devices)
        # Our loss_fn moves its inputs to its device (which is DEVICE)
        loss, loc_loss, conf_loss = loss_fn(predicted_locs, predicted_scores, gt_boxes_batch_eval_gpu, gt_labels_batch_eval_gpu)
        
        if torch.isnan(loss):
            print(f"!!! NaN validation loss detected at Epoch {epoch+1}, Batch {i_eval+1}. Skipping mAP update for this batch. !!!")
        else:
            total_loss_epoch += loss.item()
            total_loc_loss_epoch += loc_loss
            total_conf_loss_epoch += conf_loss

        # Prepare predictions and targets FOR METRIC ON CPU
        preds_for_metric_cpu = []
        targets_for_metric_cpu = []

        # Anchors for eval should be on the same device as predicted_locs for decoding
        current_anchors_cxcywh = anchors_cxcywh_for_eval.to(predicted_locs.device)


        for img_idx in range(images.size(0)): # Iterate through batch items
            # Ground Truth for metric (move to CPU)
            true_img_boxes_cpu = targets[img_idx]['boxes'].cpu() # Original targets from dataloader are on CPU
            true_img_labels_cpu = targets[img_idx]['labels'].cpu()
            
            targets_for_metric_cpu.append(
                dict(
                    boxes=true_img_boxes_cpu * IMAGE_SIZE, # Denormalize
                    labels=true_img_labels_cpu
                )
            )

            # Predictions (decode on GPU, then move results to CPU)
            img_pred_locs = predicted_locs[img_idx]         # On GPU
            img_pred_scores = predicted_scores[img_idx]     # On GPU

            decoded_cx = img_pred_locs[:, 0] * variance[0] * current_anchors_cxcywh[:, 2] + current_anchors_cxcywh[:, 0]
            decoded_cy = img_pred_locs[:, 1] * variance[1] * current_anchors_cxcywh[:, 3] + current_anchors_cxcywh[:, 1]
            decoded_w = torch.exp(img_pred_locs[:, 2] * variance[2]) * current_anchors_cxcywh[:, 2]
            decoded_h = torch.exp(img_pred_locs[:, 3] * variance[3]) * current_anchors_cxcywh[:, 3]
            
            decoded_boxes_cxcywh = torch.stack([decoded_cx, decoded_cy, decoded_w, decoded_h], dim=1)
            decoded_boxes_xyxy_normalized = cxcywh_to_xyxy(decoded_boxes_cxcywh)
            decoded_boxes_xyxy_normalized = torch.clamp(decoded_boxes_xyxy_normalized, 0, 1)

            class_probs_objects = F.softmax(img_pred_scores[:, 1:], dim=1)
            
            final_boxes_img_cpu = []
            final_scores_img_cpu = []
            final_labels_img_cpu = []
            
            conf_threshold_eval = 0.01
            nms_iou_threshold_eval = 0.45

            for class_j in range(class_probs_objects.size(1)):
                actual_class_label = class_j
                class_specific_scores = class_probs_objects[:, class_j]
                score_mask = class_specific_scores > conf_threshold_eval
                if not score_mask.any():
                    continue

                candidate_boxes_this_class = decoded_boxes_xyxy_normalized[score_mask]
                candidate_scores_this_class = class_specific_scores[score_mask]
                
                if candidate_boxes_this_class.numel() == 0:
                    continue
                
                # NMS can also be done on CPU to save GPU memory if it's a bottleneck
                # For now, NMS input boxes are on GPU, output indices are used to gather GPU tensors
                # Then results are moved to CPU.
                keep_indices = torch.ops.torchvision.nms(
                    candidate_boxes_this_class,
                    candidate_scores_this_class,
                    nms_iou_threshold_eval
                )
                
                # Move results of NMS to CPU before appending
                final_boxes_img_cpu.append((candidate_boxes_this_class[keep_indices] * IMAGE_SIZE).cpu())
                final_scores_img_cpu.append(candidate_scores_this_class[keep_indices].cpu())
                final_labels_img_cpu.append(torch.full_like(candidate_scores_this_class[keep_indices], actual_class_label, dtype=torch.long).cpu())

            if final_boxes_img_cpu:
                preds_for_metric_cpu.append(
                    dict(
                        boxes=torch.cat(final_boxes_img_cpu),
                        scores=torch.cat(final_scores_img_cpu),
                        labels=torch.cat(final_labels_img_cpu)
                    )
                )
            else:
                 preds_for_metric_cpu.append(dict(
                    boxes=torch.empty(0, 4, device="cpu"), # Explicitly CPU
                    scores=torch.empty(0, device="cpu"),
                    labels=torch.empty(0, dtype=torch.long, device="cpu")
                ))
        
        try:
            metric.update(preds_for_metric_cpu, targets_for_metric_cpu) # UPDATE WITH CPU TENSORS
        except Exception as e:
            print(f"Error updating metric: {e}. Preds count: {len(preds_for_metric_cpu)}, Targets count: {len(targets_for_metric_cpu)}")
            # ... error printing ...
        
        # Clear GPU cache more frequently during eval if memory is very tight
        if device == "cuda":
            del predicted_locs, predicted_scores, images # Delete GPU tensors from this batch
            del gt_boxes_batch_eval_gpu, gt_labels_batch_eval_gpu
            torch.cuda.empty_cache()


    num_batches_val = len(data_loader)
    if num_batches_val == 0: # Should not happen with valid dataset
        print("Warning: val_loader was empty. No validation occurred.")
        return 0.0, 0.0, 0.0, 0.0, 0.0

    avg_loss_val = total_loss_epoch / num_batches_val if num_batches_val > 0 else 0.0
    avg_loc_loss_val = total_loc_loss_epoch / num_batches_val if num_batches_val > 0 else 0.0
    avg_conf_loss_val = total_conf_loss_epoch / num_batches_val if num_batches_val > 0 else 0.0
    print(f"Validation Average Loss: {avg_loss_val:.4f} (Loc: {avg_loc_loss_val:.4f} Conf: {avg_conf_loss_val:.4f})")

    # metric.compute() will now run on CPU as its state was updated with CPU tensors
    map_results = {'map': torch.tensor(0.0), 'map_50': torch.tensor(0.0), 'map_75': torch.tensor(0.0)}
    try:
        # Ensure metric itself is on CPU if compute implies device from state
        # metric_cpu = metric.cpu() # or metric.to("cpu")
        # computed_metrics = metric_cpu.compute()
        computed_metrics = metric.compute() # Should work if update was with CPU tensors
        map_results.update({k: v.cpu() for k,v in computed_metrics.items()}) # Ensure results are CPU scalars
        print(f"Validation mAP: {map_results['map'].item():.4f}")
        print(f"Validation mAP_50: {map_results['map_50'].item():.4f}")
        print(f"Validation mAP_75: {map_results['map_75'].item():.4f}")
    except Exception as e:
        print(f"Could not compute mAP: {e}")

    epoch_time = time.time() - start_time
    print(f"Validation time taken: {epoch_time:.2f}s")
    return avg_loss_val, avg_loc_loss_val, avg_conf_loss_val, map_results['map_50'].item(), map_results['map'].item()


def main():
    print(f"Using device: {DEVICE}")
    model_dir = os.path.dirname(MODEL_SAVE_PATH)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Created models directory: {model_dir}")

    train_transform = SSDTransform(image_size=IMAGE_SIZE, train=True)
    val_transform = SSDTransform(image_size=IMAGE_SIZE, train=False)

    dataset_year = '2007' 
    train_image_set = 'trainval' if dataset_year == '2007' else 'train'
    val_image_set = 'test' if dataset_year == '2007' else 'val'
    
    try:
        train_dataset = PascalVOCDataset(root=DATA_DIR, year=dataset_year, image_set=train_image_set, transforms=train_transform)
        val_dataset = PascalVOCDataset(root=DATA_DIR, year=dataset_year, image_set=val_image_set, transforms=val_transform)
    except Exception as e:
        print(f"Error initializing dataset: {e}. Ensure dataset is downloaded or path is correct.")
        print(f"DATA_DIR used: {DATA_DIR}")
        return

    num_workers_setting = 0 # Forcing 0 for max stability, especially on Windows / limited memory
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=num_workers_setting, pin_memory=False) # pin_memory False with num_workers=0
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers_setting, pin_memory=False)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("One of the datasets is empty. Please check dataset paths and content.")
        return
        
    print(f"Training on {len(train_dataset)} images ({len(train_loader)} batches), validating on {len(val_dataset)} images ({len(val_loader)} batches).")

    model = SSDResNet50(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, pretrained_backbone=True, backbone_requires_grad=True)
    model.to(DEVICE)
    
    # Anchors for evaluation should be on the same device as predictions when decoding happens.
    # `current_anchors_cxcywh` inside evaluate() handles this.
    anchors_for_eval_param_to_evaluate_func = model.anchors_cxcywh 

    loss_fn = MultiBoxLoss(
        anchors_cxcywh=model.anchors_cxcywh, # These are already on DEVICE from model init
        num_classes=NUM_CLASSES,
        device=DEVICE # Loss function will operate on DEVICE
    )

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    except TypeError:
        print("Warning: ReduceLROnPlateau verbose argument not supported. Creating scheduler without it.")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    best_map50 = 0.0

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS): # NUM_EPOCHS from config
        train_loss, train_loc_loss, train_conf_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, epoch, DEVICE, grad_clip_norm=1.0
        )
        
        val_loss, val_loc_loss, val_conf_loss, current_val_map50, current_val_map = evaluate(
            model, val_loader, loss_fn, DEVICE, epoch, anchors_for_eval_param_to_evaluate_func, NUM_CLASSES
        )
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_val_map50)
        elif scheduler is not None: # For other schedulers like StepLR
            scheduler.step()

        if current_val_map50 > best_map50:
            best_map50 = current_val_map50
            print(f"Epoch {epoch+1}: New best model with mAP@0.50: {current_val_map50:.4f}. Saving model...")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'map50': current_val_map50,
                'map': current_val_map
            }, MODEL_SAVE_PATH)
            print(f"Model saved to {MODEL_SAVE_PATH}")
        else:
            print(f"Epoch {epoch+1}: mAP@0.50 ({current_val_map50:.4f}) did not improve from best ({best_map50:.4f}). Not saving.")

        
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

    print("Training finished.")
    if os.path.exists(MODEL_SAVE_PATH) and best_map50 > 0: # Check if a model was actually saved
        print(f"Best validation mAP@0.50 achieved: {best_map50:.4f}")
        print(f"Best model saved to {MODEL_SAVE_PATH}")
    else:
        print("No model improved enough to be saved, or training was too short.")


if __name__ == '__main__':
    main()