# src/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import cxcywh_to_xyxy, xyxy_to_cxcywh # For box conversions
# generate_anchor_boxes is not directly used here, anchors come from model

# --- IoU (Intersection over Union) or Jaccard Index ---
def find_jaccard_overlap(boxes1_xyxy, boxes2_xyxy):
    """
    Calculate the Intersection over Union (IoU) of two sets of boxes.
    Args:
        boxes1_xyxy (Tensor): (N, 4) representing N boxes in [xmin, ymin, xmax, ymax] format.
        boxes2_xyxy (Tensor): (M, 4) representing M boxes in [xmin, ymin, xmax, ymax] format.
    Returns:
        iou (Tensor): (N, M) representing IoU scores for all pairs of boxes.
    """
    # Ensure boxes are on the same device
    if boxes1_xyxy.device != boxes2_xyxy.device:
        # This might happen if one set of boxes (e.g., anchors) is on a different device
        # than the other (e.g., ground truth moved to a specific device).
        # It's generally better to ensure inputs are on the correct device before calling.
        # However, for robustness within this function:
        # Let's assume boxes1_xyxy dictates the device if they differ.
        boxes2_xyxy = boxes2_xyxy.to(boxes1_xyxy.device)


    boxes1_expanded = boxes1_xyxy.unsqueeze(1) # (N, 1, 4)
    boxes2_expanded = boxes2_xyxy.unsqueeze(0) # (1, M, 4)

    inter_xmin = torch.max(boxes1_expanded[..., 0], boxes2_expanded[..., 0])
    inter_ymin = torch.max(boxes1_expanded[..., 1], boxes2_expanded[..., 1])
    inter_xmax = torch.min(boxes1_expanded[..., 2], boxes2_expanded[..., 2])
    inter_ymax = torch.min(boxes1_expanded[..., 3], boxes2_expanded[..., 3])

    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
    intersection_area = inter_w * inter_h

    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection_area
    
    iou = intersection_area / (union_area + 1e-7) # Increased epsilon slightly for stability
    
    return iou


# --- SSD MultiBox Loss ---
class MultiBoxLoss(nn.Module):
    def __init__(self, anchors_cxcywh, num_classes, iou_threshold=0.5, neg_pos_ratio=3, device='cpu'):
        """
        Args:
            anchors_cxcywh (Tensor): Prior/default/anchor boxes from the model,
                                     shape (num_total_anchors, 4) in (cx, cy, w, h) format,
                                     normalized by image dimensions. Expected to be on the correct device.
            num_classes (int): Number of classes including background.
            iou_threshold (float): Threshold for matching anchors to ground truth.
            neg_pos_ratio (int): Ratio of negative to positive samples for hard negative mining.
            device (str): 'cuda' or 'cpu'. This loss module will work on this device.
        """
        super().__init__()
        self.device = device # Store the target device

        # Ensure anchors are on the correct device for this loss module.
        # The anchors_cxcywh passed from the model should already be on the model's device.
        # We move them to self.device if they are not already, to be sure.
        self.anchors_cxcywh = anchors_cxcywh.to(self.device)
        self.anchors_xyxy = cxcywh_to_xyxy(self.anchors_cxcywh).to(self.device) # Also ensure this is on self.device

        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_locs, predicted_scores, gt_boxes_batch, gt_labels_batch):
        """
        Args:
            predicted_locs (Tensor): Predicted locations from SSD, (batch_size, num_anchors, 4). On self.device.
            predicted_scores (Tensor): Predicted class scores from SSD, (batch_size, num_anchors, num_classes). On self.device.
            gt_boxes_batch (list of Tensors): Ground truth boxes. Each tensor (num_obj, 4) in [xmin, ymin, xmax, ymax].
                                           Needs to be moved to self.device.
            gt_labels_batch (list of Tensors): Ground truth labels. Each tensor (num_obj,). Needs to be moved to self.device.
        Returns:
            loss (Tensor), loc_loss_val (float), conf_loss_val (float)
        """
        batch_size = predicted_locs.size(0)
        num_anchors_total = self.anchors_cxcywh.size(0)

        # Ensure predictions are on self.device (they should be if model is on self.device)
        predicted_locs = predicted_locs.to(self.device)
        predicted_scores = predicted_scores.to(self.device)

        true_locs_batch = torch.zeros((batch_size, num_anchors_total, 4), dtype=torch.float32, device=self.device)
        true_classes_batch = torch.zeros((batch_size, num_anchors_total), dtype=torch.int64, device=self.device)
        positive_anchor_mask_batch = torch.zeros((batch_size, num_anchors_total), dtype=torch.bool, device=self.device)

        for i in range(batch_size):
            gt_boxes = gt_boxes_batch[i].to(self.device)
            gt_labels = gt_labels_batch[i].to(self.device)
            
            if gt_boxes.numel() == 0:
                continue

            # self.anchors_xyxy is already on self.device from __init__
            iou_scores = find_jaccard_overlap(gt_boxes, self.anchors_xyxy) # (num_objects, num_anchors)
            # All tensors involved here (gt_boxes, self.anchors_xyxy) are now on self.device.
            # So, iou_scores and subsequent derived tensors will also be on self.device.

            gt_best_anchor_iou, gt_best_anchor_idx = iou_scores.max(dim=1)
            anchor_best_gt_iou, anchor_best_gt_idx = iou_scores.max(dim=0)

            positive_mask_candidate = anchor_best_gt_iou > self.iou_threshold
            
            for obj_idx in range(gt_boxes.size(0)):
                if gt_best_anchor_iou[obj_idx] > 1e-5:
                    best_anchor_for_this_gt = gt_best_anchor_idx[obj_idx]
                    positive_mask_candidate[best_anchor_for_this_gt] = True
                    # Update the gt assignment for this anchor if this gt offers a better IoU or if it's the primary match
                    if anchor_best_gt_iou[best_anchor_for_this_gt] < gt_best_anchor_iou[obj_idx]:
                         anchor_best_gt_idx[best_anchor_for_this_gt] = obj_idx


            # positive_anchors_indices will be on self.device
            positive_anchors_indices = torch.where(positive_mask_candidate)[0]
            
            if positive_anchors_indices.numel() > 0: # Only proceed if there are positive anchors
                positive_anchor_mask_batch[i, positive_anchors_indices] = True

                matched_gt_boxes_for_positive_anchors = gt_boxes[anchor_best_gt_idx[positive_anchors_indices]]
                matched_gt_labels_for_positive_anchors = gt_labels[anchor_best_gt_idx[positive_anchors_indices]]

                true_classes_batch[i, positive_anchors_indices] = matched_gt_labels_for_positive_anchors + 1
                
                # self.anchors_cxcywh is on self.device from __init__
                # positive_anchors_indices is on self.device
                # So, positive_anchors_cxcywh will be on self.device
                positive_anchors_cxcywh = self.anchors_cxcywh[positive_anchors_indices] # This was the error line
                
                matched_gt_boxes_cxcywh = xyxy_to_cxcywh(matched_gt_boxes_for_positive_anchors)

                target_locs_cx = (matched_gt_boxes_cxcywh[:, 0] - positive_anchors_cxcywh[:, 0]) / positive_anchors_cxcywh[:, 2]
                target_locs_cy = (matched_gt_boxes_cxcywh[:, 1] - positive_anchors_cxcywh[:, 1]) / positive_anchors_cxcywh[:, 3]
                target_locs_w = torch.log(matched_gt_boxes_cxcywh[:, 2] / positive_anchors_cxcywh[:, 2] + 1e-7) # Add epsilon for log
                target_locs_h = torch.log(matched_gt_boxes_cxcywh[:, 3] / positive_anchors_cxcywh[:, 3] + 1e-7) # Add epsilon for log
                
                true_locs_batch[i, positive_anchors_indices, 0] = target_locs_cx
                true_locs_batch[i, positive_anchors_indices, 1] = target_locs_cy
                true_locs_batch[i, positive_anchors_indices, 2] = target_locs_w
                true_locs_batch[i, positive_anchors_indices, 3] = target_locs_h

        # Localization Loss
        # positive_anchor_mask_batch is on self.device
        # predicted_locs and true_locs_batch are on self.device
        # So loc_loss will be on self.device
        loc_loss = self.smooth_l1_loss(
            predicted_locs[positive_anchor_mask_batch],
            true_locs_batch[positive_anchor_mask_batch]
        )

        # Confidence Loss
        num_positive_anchors_batch_per_image = positive_anchor_mask_batch.sum(dim=1)
        
        conf_loss_all = self.cross_entropy_loss(
            predicted_scores.view(-1, self.num_classes),
            true_classes_batch.view(-1)
        )
        conf_loss_all = conf_loss_all.view(batch_size, num_anchors_total)

        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_anchor_mask_batch] = 0 # Zero out loss for positive anchors for mining
        
        # Hard Negative Mining
        # Sort negative losses in descending order for each image in the batch
        # conf_loss_neg_sorted_values, conf_loss_neg_sorted_indices = conf_loss_neg.sort(dim=1, descending=True)
        
        # Calculate number of negatives to keep for each image
        num_negatives_to_keep_per_image = torch.min(
            self.neg_pos_ratio * num_positive_anchors_batch_per_image,
            num_anchors_total - num_positive_anchors_batch_per_image
        )
        num_negatives_to_keep_per_image = torch.clamp(num_negatives_to_keep_per_image, min=0)

        hard_negative_mask_batch = torch.zeros_like(positive_anchor_mask_batch, dtype=torch.bool) # on self.device

        for i in range(batch_size):
            num_neg_to_select = num_negatives_to_keep_per_image[i].item()
            if num_neg_to_select > 0:
                # Select from anchors that are NOT positive
                image_negative_losses = conf_loss_neg[i].clone() # Losses for this image, positives are 0
                # We need indices of the actual anchors, not sorted values
                # Get losses only for actual negative anchors for this image
                current_negative_mask = ~positive_anchor_mask_batch[i]
                losses_of_actual_negatives = image_negative_losses[current_negative_mask]

                if losses_of_actual_negatives.numel() > 0 : # if there are any negative anchors
                    num_neg_to_select = min(num_neg_to_select, losses_of_actual_negatives.numel()) # Don't try to select more than available
                    if num_neg_to_select > 0:
                        _, top_k_indices_within_actual_negatives = torch.topk(losses_of_actual_negatives, k=num_neg_to_select)
                        
                        # Map these indices back to original anchor indices
                        original_indices_of_all_anchors = torch.arange(num_anchors_total, device=self.device)
                        original_indices_of_actual_negatives = original_indices_of_all_anchors[current_negative_mask]
                        hard_neg_original_indices = original_indices_of_actual_negatives[top_k_indices_within_actual_negatives]
                        
                        hard_negative_mask_batch[i, hard_neg_original_indices] = True
            elif num_positive_anchors_batch_per_image[i] == 0: # No positives, so num_neg_to_select was 0
                # If no positives, select a small fixed number of hardest negatives among all anchors
                # (as all are technically negative)
                num_default_neg = min(self.neg_pos_ratio * 20, num_anchors_total) # e.g., up to 3*20=60 negatives
                if conf_loss_neg[i].numel() > 0 and num_default_neg > 0:
                     num_default_neg = min(num_default_neg, conf_loss_neg[i].numel())
                     if num_default_neg > 0:
                        _, top_k_default_neg_indices = torch.topk(conf_loss_neg[i], k=num_default_neg) # conf_loss_neg already has 0 for positives (if any)
                        hard_negative_mask_batch[i, top_k_default_neg_indices] = True

        final_conf_loss_mask = positive_anchor_mask_batch | hard_negative_mask_batch
        conf_loss_sum = conf_loss_all[final_conf_loss_mask].sum()

        num_total_positive_anchors_in_batch = positive_anchor_mask_batch.sum()
        
        if num_total_positive_anchors_in_batch > 0:
            total_loss = (loc_loss + conf_loss_sum) / num_total_positive_anchors_in_batch
            loc_loss_val = loc_loss.item() / num_total_positive_anchors_in_batch
            conf_loss_val = conf_loss_sum.item() / num_total_positive_anchors_in_batch
        else:
            # If no positive anchors, loc_loss is 0.
            # Confidence loss is sum over hard negatives. Normalize by batch_size or num_hard_negatives.
            num_hard_negatives_in_batch = hard_negative_mask_batch.sum()
            if num_hard_negatives_in_batch > 0:
                total_loss = conf_loss_sum / num_hard_negatives_in_batch
                conf_loss_val = total_loss.item() # conf_loss_sum.item() / num_hard_negatives_in_batch
            elif conf_loss_sum > 1e-7 : # Some residual loss from non-selected negatives if logic changes
                total_loss = conf_loss_sum / batch_size # Fallback, less ideal
                conf_loss_val = total_loss.item()
            else: # No positives, no hard negatives selected with significant loss
                total_loss = torch.tensor(0., device=self.device)
                conf_loss_val = 0.0
            loc_loss_val = 0.0

        return total_loss, loc_loss_val, conf_loss_val

# Test the loss function
if __name__ == '__main__':
    from src.config import NUM_CLASSES, DEVICE, IMAGE_SIZE, VOC_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
    from src.model import SSDResNet50

    print("Testing MultiBoxLoss...")
    
    try:
        print("Initializing dummy model for anchors...")
        # Make sure IMAGE_SIZE is correctly passed if model uses it for anchor gen internally
        temp_model = SSDResNet50(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, backbone_requires_grad=False)
        temp_model.to(DEVICE) # Model should be on DEVICE
        anchors_cxcywh_from_model = temp_model.anchors_cxcywh # These are already on DEVICE from model init
        print(f"Using {anchors_cxcywh_from_model.shape[0]} anchors from model, on device: {anchors_cxcywh_from_model.device}")
    except Exception as e:
        print(f"Could not initialize model for anchors: {e}")
        print("Using random dummy anchors for loss test.")
        num_dummy_anchors = 8732 
        anchors_cxcywh_from_model = torch.rand(num_dummy_anchors, 4, device=DEVICE) # Create on DEVICE
        anchors_cxcywh_from_model[:, 2:] = torch.clamp(anchors_cxcywh_from_model[:, 2:], min=0.01, max=1.0)
        anchors_cxcywh_from_model[:, :2] = torch.clamp(anchors_cxcywh_from_model[:, :2], min=0.0, max=1.0)

    loss_fn = MultiBoxLoss(
        anchors_cxcywh=anchors_cxcywh_from_model, # Pass anchors already on DEVICE
        num_classes=NUM_CLASSES, 
        iou_threshold=0.5, 
        neg_pos_ratio=3,
        device=DEVICE # Loss function will operate on this device
    )

    batch_size = 2
    num_anchors = anchors_cxcywh_from_model.size(0)

    pred_locs = torch.randn(batch_size, num_anchors, 4).to(DEVICE)
    pred_scores = torch.randn(batch_size, num_anchors, NUM_CLASSES).to(DEVICE)

    gt_boxes1 = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]], dtype=torch.float32) # CPU
    gt_labels1 = torch.tensor([CLASS_TO_IDX["cat"], CLASS_TO_IDX["dog"]], dtype=torch.int64) # CPU, using CLASS_TO_IDX
    gt_boxes2 = torch.tensor([[0.2, 0.3, 0.4, 0.7]], dtype=torch.float32) # CPU
    gt_labels2 = torch.tensor([CLASS_TO_IDX["person"]], dtype=torch.int64) # CPU

    # In the actual training loop, gt_boxes_batch and gt_labels_batch elements are moved to DEVICE.
    # For this test, we'll mimic that by preparing them on CPU and relying on the loss_fn.forward to move them.
    gt_boxes_list = [gt_boxes1, gt_boxes2]
    gt_labels_list = [gt_labels1, gt_labels2]

    print("\n--- Test Case 1: With objects ---")
    total_loss, loc_loss, conf_loss = loss_fn(pred_locs, pred_scores, gt_boxes_list, gt_labels_list)
    print(f"Total Loss: {total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss}")
    print(f"Localization Loss: {loc_loss}")
    print(f"Confidence Loss: {conf_loss}")
    if isinstance(total_loss, torch.Tensor):
      assert not torch.isnan(total_loss), "Loss is NaN!"
      assert total_loss.item() >= -1e-5, "Loss is significantly negative!" # allow for small float inaccuracies

    print("\n--- Test Case 2: Image with no objects ---")
    gt_boxes_no_obj_list = [torch.empty((0,4), dtype=torch.float32), gt_boxes2]
    gt_labels_no_obj_list = [torch.empty((0,), dtype=torch.int64), gt_labels2]
    
    total_loss_no_obj, loc_loss_no_obj, conf_loss_no_obj = loss_fn(pred_locs, pred_scores, gt_boxes_no_obj_list, gt_labels_no_obj_list)
    print(f"Total Loss (no obj): {total_loss_no_obj.item() if isinstance(total_loss_no_obj, torch.Tensor) else total_loss_no_obj}")
    print(f"Localization Loss (no obj): {loc_loss_no_obj}")
    print(f"Confidence Loss (no obj): {conf_loss_no_obj}")
    if isinstance(total_loss_no_obj, torch.Tensor):
      assert not torch.isnan(total_loss_no_obj), "Loss (no obj) is NaN!"

    print("\n--- Test Case 3: Batch with no objects at all ---")
    gt_boxes_all_no_obj_list = [torch.empty((0,4), dtype=torch.float32), torch.empty((0,4), dtype=torch.float32)]
    gt_labels_all_no_obj_list = [torch.empty((0,), dtype=torch.int64), torch.empty((0,), dtype=torch.int64)]

    total_loss_all_no, loc_loss_all_no, conf_loss_all_no = loss_fn(pred_locs, pred_scores, gt_boxes_all_no_obj_list, gt_labels_all_no_obj_list)
    print(f"Total Loss (all no obj): {total_loss_all_no.item() if isinstance(total_loss_all_no, torch.Tensor) else total_loss_all_no}")
    print(f"Localization Loss (all no obj): {loc_loss_all_no}")
    print(f"Confidence Loss (all no obj): {conf_loss_all_no}")
    if isinstance(total_loss_all_no, torch.Tensor):
      assert not torch.isnan(total_loss_all_no), "Loss (all no obj) is NaN!"
    
    print("\nMultiBoxLoss test finished.")