import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from src.model import SSDResNet50
from src.config import (
    DEVICE, NUM_CLASSES, IMAGE_SIZE, MODEL_SAVE_PATH,
    VOC_CLASSES, IDX_TO_CLASS # Make sure IDX_TO_CLASS is defined in config.py
)
from src.utils import cxcywh_to_xyxy
import torch.nn.functional as F
from torchvision.ops import nms # For Non-Maximum Suppression

# Ensure IDX_TO_CLASS is available (add to config.py if not already)
# In config.py:
# IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(VOC_CLASSES)}

def load_model(model_path, num_classes, device):
    """Loads the trained SSD model."""
    model = SSDResNet50(num_classes=num_classes, image_size=IMAGE_SIZE)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle potential DataParallel wrapper if model was trained on multiple GPUs
    # and saved with module. prefix
    state_dict = checkpoint['model_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove `module.`
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}, epoch {checkpoint.get('epoch', 'N/A')}")
    return model

def preprocess_image(image_path, image_size):
    """Loads and preprocesses an image for inference."""
    image = Image.open(image_path).convert("RGB")
    
    # Store original size for later rescaling of boxes
    original_width, original_height = image.size
    
    # Transform similar to validation transform but without target
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0) # Add batch dimension
    return img_tensor, original_width, original_height

def decode_predictions(predicted_locs_single, predicted_scores_single, anchors_cxcywh,
                       confidence_threshold=0.5, nms_iou_threshold=0.45, top_k=200):
    """
    Decodes raw model predictions for a single image into bounding boxes, labels, and scores.
    Args:
        predicted_locs_single (Tensor): (num_anchors, 4) predicted location offsets.
        predicted_scores_single (Tensor): (num_anchors, num_classes) predicted class scores (logits).
        anchors_cxcywh (Tensor): (num_anchors, 4) anchor boxes in (cx, cy, w, h) format.
        confidence_threshold (float): Minimum score to consider a detection.
        nms_iou_threshold (float): IoU threshold for NMS.
        top_k (int): Keep at most top_k detections after NMS.
    Returns:
        final_boxes (Tensor): (num_detections, 4) [xmin, ymin, xmax, ymax] pixel coordinates.
        final_labels (Tensor): (num_detections,) predicted class indices.
        final_scores (Tensor): (num_detections,) confidence scores.
    """
    device = predicted_locs_single.device
    num_anchors = anchors_cxcywh.size(0)
    num_classes = predicted_scores_single.size(1) # Includes background

    # 1. Apply offsets to anchors to get decoded bounding box coordinates
    variance = torch.tensor([0.1, 0.1, 0.2, 0.2], device=device) # Standard SSD variances
    
    decoded_cx = predicted_locs_single[:, 0] * variance[0] * anchors_cxcywh[:, 2] + anchors_cxcywh[:, 0]
    decoded_cy = predicted_locs_single[:, 1] * variance[1] * anchors_cxcywh[:, 3] + anchors_cxcywh[:, 1]
    decoded_w = torch.exp(predicted_locs_single[:, 2] * variance[2]) * anchors_cxcywh[:, 2]
    decoded_h = torch.exp(predicted_locs_single[:, 3] * variance[3]) * anchors_cxcywh[:, 3]
    
    decoded_boxes_cxcywh = torch.stack([decoded_cx, decoded_cy, decoded_w, decoded_h], dim=1)
    # Convert to [xmin, ymin, xmax, ymax] format, still normalized [0,1]
    decoded_boxes_xyxy_normalized = cxcywh_to_xyxy(decoded_boxes_cxcywh)
    # Clip to [0, 1] image boundaries
    decoded_boxes_xyxy_normalized = torch.clamp(decoded_boxes_xyxy_normalized, 0, 1)

    # 2. Get class probabilities (apply softmax to scores, ignoring background class for object detection)
    # Background class is assumed to be index 0. Object classes are 1 to num_classes-1.
    class_probs = F.softmax(predicted_scores_single[:, 1:], dim=1) # (num_anchors, num_object_classes)

    all_boxes = []
    all_scores = []
    all_labels = []

    # 3. Perform NMS for each class separately
    for class_idx in range(class_probs.size(1)): # Iterate over object classes (0 to num_obj_classes-1)
        actual_class_label = class_idx # This is the 0-19 VOC label. If using IDX_TO_CLASS, this matches.
                                       # If class_probs was from scores[:,0:], then actual_class_label needs adjustment.

        class_specific_scores = class_probs[:, class_idx] # Scores for this class for all anchors

        # Filter by confidence threshold
        score_above_threshold_mask = class_specific_scores > confidence_threshold
        if not score_above_threshold_mask.any():
            continue

        class_boxes_candidate = decoded_boxes_xyxy_normalized[score_above_threshold_mask]
        class_scores_candidate = class_specific_scores[score_above_threshold_mask]
        
        # Perform NMS
        # torchvision.ops.nms expects boxes in [x1, y1, x2, y2] format (not normalized here, but works if consistent)
        # It also expects scores.
        keep_indices = nms(class_boxes_candidate, class_scores_candidate, nms_iou_threshold)
        
        final_class_boxes = class_boxes_candidate[keep_indices]
        final_class_scores = class_scores_candidate[keep_indices]
        final_class_labels = torch.full_like(final_class_scores, actual_class_label, dtype=torch.long)
        
        all_boxes.append(final_class_boxes)
        all_scores.append(final_class_scores)
        all_labels.append(final_class_labels)

    if not all_boxes: # No detections after NMS for any class
        return torch.empty(0, 4), torch.empty(0, dtype=torch.long), torch.empty(0)

    # Concatenate results from all classes
    final_boxes_cat = torch.cat(all_boxes, dim=0)
    final_scores_cat = torch.cat(all_scores, dim=0)
    final_labels_cat = torch.cat(all_labels, dim=0)

    # Keep only top_k detections overall (optional, if too many detections)
    if final_scores_cat.size(0) > top_k:
        sorted_scores, sorted_indices = torch.sort(final_scores_cat, descending=True)
        top_k_indices = sorted_indices[:top_k]
        final_boxes_cat = final_boxes_cat[top_k_indices]
        final_scores_cat = final_scores_cat[top_k_indices]
        final_labels_cat = final_labels_cat[top_k_indices]
        
    return final_boxes_cat, final_labels_cat, final_scores_cat


def visualize_predictions(image_path, boxes_normalized, labels, scores, idx_to_class,
                          original_width, original_height, output_path="predicted_image.jpg"):
    """Draws bounding boxes on the image and saves it."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Scale normalized boxes to original image dimensions
    # boxes_normalized is [xmin, ymin, xmax, ymax] in [0,1] range
    boxes_pixel = boxes_normalized.clone()
    boxes_pixel[:, [0, 2]] *= original_width
    boxes_pixel[:, [1, 3]] *= original_height
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for i in range(boxes_pixel.size(0)):
        box = boxes_pixel[i].tolist()
        label_idx = labels[i].item()
        score = scores[i].item()
        
        class_name = idx_to_class.get(label_idx, "Unknown")
        display_text = f"{class_name}: {score:.2f}"
        
        # Draw rectangle
        draw.rectangle(box, outline="red", width=3)
        
        # Draw text background
        text_bbox = draw.textbbox((box[0], box[1]), display_text, font=font) # For PIL > 9.2.0
        # For older PIL: text_size = draw.textsize(display_text, font=font)
        # text_bbox = (box[0], box[1], box[0] + text_size[0], box[1] + text_size[1])
        
        # Adjust text position if it goes out of image
        text_x = box[0]
        text_y = box[1] - (text_bbox[3] - text_bbox[1]) # text_height
        if text_y < 0:
            text_y = box[1] + 1 # Place below top-left if it overflows

        draw.rectangle(text_bbox, fill="red")
        draw.text((text_x, text_y), display_text, fill="white", font=font)
        
    image.save(output_path)
    print(f"Prediction visualized and saved to {output_path}")
    # image.show() # Optionally display the image

if __name__ == '__main__':
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model checkpoint not found at {MODEL_SAVE_PATH}. Please train the model first.")
        exit()

    # --- Configuration for Prediction ---
    image_path_to_predict = "generated_swag.png" # <--- CHANGE THIS TO YOUR IMAGE PATH
    # Example: Create a dummy image if you don't have one
    if not os.path.exists(image_path_to_predict) or image_path_to_predict == "path/to/your/image.jpg":
        print(f"Warning: Image path '{image_path_to_predict}' not found or is default.")
        try:
            dummy_img = Image.new('RGB', (600, 400), color = 'skyblue')
            dummy_img_path = "dummy_test_image.jpg"
            dummy_img.save(dummy_img_path)
            image_path_to_predict = dummy_img_path
            print(f"Using a dummy image: {dummy_img_path}")
        except Exception as e:
            print(f"Could not create dummy image: {e}. Please provide a valid image_path_to_predict.")
            exit()
            
    confidence_thresh = 0.3 # Lower for more detections, higher for fewer but more confident ones
    nms_thresh = 0.45
    output_image_file = "prediction_output.jpg"
    # --- End Configuration ---

    # 1. Load Model
    model = load_model(MODEL_SAVE_PATH, NUM_CLASSES, DEVICE)
    anchors_cxcywh = model.anchors_cxcywh.to(DEVICE) # Get anchors from the model

    # 2. Preprocess Image
    img_tensor, orig_w, orig_h = preprocess_image(image_path_to_predict, IMAGE_SIZE)
    img_tensor = img_tensor.to(DEVICE)

    # 3. Make Prediction
    with torch.no_grad():
        predicted_locs, predicted_scores = model(img_tensor)

    # Output is for a batch of size 1, so take the first element
    predicted_locs_single = predicted_locs[0]     # (num_anchors, 4)
    predicted_scores_single = predicted_scores[0] # (num_anchors, num_classes)

    # 4. Decode Predictions
    final_boxes_norm, final_labels, final_scores = decode_predictions(
        predicted_locs_single, predicted_scores_single, anchors_cxcywh,
        confidence_threshold=confidence_thresh, nms_iou_threshold=nms_thresh
    )
    
    print(f"\nDetected {final_boxes_norm.size(0)} objects:")
    for i in range(final_boxes_norm.size(0)):
        label_name = IDX_TO_CLASS.get(final_labels[i].item(), "Unknown")
        print(f"  - {label_name}: {final_scores[i].item():.3f} at {final_boxes_norm[i].cpu().numpy()}")

    # 5. Visualize (if any detections)
    if final_boxes_norm.size(0) > 0:
        visualize_predictions(image_path_to_predict, final_boxes_norm.cpu(), 
                              final_labels.cpu(), final_scores.cpu(), 
                              IDX_TO_CLASS, orig_w, orig_h, output_path=output_image_file)
    else:
        print("No objects detected with the current thresholds.")
        # Save original image if no detections, or indicate no detections
        Image.open(image_path_to_predict).convert("RGB").save(output_image_file)
        print(f"Original image (no detections) saved to {output_image_file}")

    if image_path_to_predict == "dummy_test_image.jpg":
        os.remove("dummy_test_image.jpg") # Clean up dummy image