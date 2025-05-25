import torch
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import os
import numpy as np
from src.config import VOC_CLASSES, CLASS_TO_IDX, IMAGE_SIZE, DATA_DIR # Assuming config.py is in src

# Global mapping for convenience
LABEL_MAP = {k: v + 1 for v, k in enumerate(VOC_CLASSES)} # +1 because 0 is background
LABEL_MAP['background'] = 0
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()} # For visualization

class PascalVOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, transforms=None, keep_difficult=False):
        super().__init__(root=root, year=year, image_set=image_set, download=True)
        self.transforms = transforms
        self.keep_difficult = keep_difficult
        self.class_to_idx = CLASS_TO_IDX # Use from config

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

        boxes = []
        labels = []
        difficulties = []

        for obj in target['annotation']['object']:
            difficult = int(obj['difficult'])
            if not self.keep_difficult and difficult:
                continue

            # Bounding box coordinates are 1-based in XML, convert to 0-based
            xmin = float(obj['bndbox']['xmin']) - 1
            ymin = float(obj['bndbox']['ymin']) - 1
            xmax = float(obj['bndbox']['xmax']) - 1
            ymax = float(obj['bndbox']['ymax']) - 1
            
            # Ensure box coordinates are valid
            if xmax <= xmin or ymax <= ymin:
                # print(f"Warning: Invalid box {obj['name']} in {self.images[index]}: {[xmin, ymin, xmax, ymax]}")
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            
            # Map class name to index
            try:
                labels.append(self.class_to_idx[obj['name']])
            except KeyError:
                # print(f"Warning: Unknown class {obj['name']} in {self.images[index]}. Skipping object.")
                boxes.pop() # Remove the last added box if label is unknown
                continue


            difficulties.append(difficult)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # If no valid objects, create empty tensors of the correct shape for boxes,
        # and ensure labels are LongTensor.
        if boxes.shape[0] == 0:
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        difficulties = torch.as_tensor(difficulties, dtype=torch.uint8)

        target_dict = {}
        target_dict['boxes'] = boxes
        target_dict['labels'] = labels
        target_dict['difficulties'] = difficulties
        target_dict['image_id'] = torch.tensor([index]) # Useful for evaluation

        original_img_width, original_img_height = img.size

        if self.transforms:
            # The transforms need to handle both image and bounding boxes
            # For now, let's assume a transform that only affects the image for simplicity in __getitem__
            # and handle box scaling in the collate_fn or a custom transform object
            img_transformed, target_dict_transformed = self.transforms(img, target_dict)
            img = img_transformed
            target_dict = target_dict_transformed


        return img, target_dict

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        # This is a simplified parser. You might need a more robust one.
        voc_dict = {}
        children = list(node)
        if children:
            obj_list = []
            for _, child in enumerate(children):
                if child.tag == 'object':
                    obj_dict = {}
                    for _, o_child in enumerate(list(child)):
                        if o_child.tag == 'bndbox':
                            bbox_dict = {}
                            for _, b_child in enumerate(list(o_child)):
                                bbox_dict[b_child.tag] = b_child.text
                            obj_dict[o_child.tag] = bbox_dict
                        else:
                            obj_dict[o_child.tag] = o_child.text
                    obj_list.append(obj_dict)
                    voc_dict[child.tag] = obj_list
                else:
                    voc_dict[child.tag] = child.text # e.g. filename, size
        # This parsing logic might need to be more robust depending on XML structure variations
        # For simplicity, assuming a flat structure for 'object' tags
        if 'object' not in voc_dict: # Handle cases with no objects
             voc_dict['object'] = []
        return {'annotation': voc_dict}


# Transformations
# For object detection, transformations need to be applied to both image and bounding boxes.
class SSDTransform:
    def __init__(self, image_size, train=True):
        self.image_size = image_size
        self.train = train
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img, target):
        original_width, original_height = img.size
        
        # Resize image
        img_resized = T.Resize((self.image_size, self.image_size))(img)
        
        # Scale bounding boxes
        boxes = target['boxes']
        if boxes.numel() > 0: # Check if there are any boxes
            # Scale factors
            scale_x = self.image_size / original_width
            scale_y = self.image_size / original_height
            
            boxes[:, 0] = boxes[:, 0] * scale_x
            boxes[:, 1] = boxes[:, 1] * scale_y
            boxes[:, 2] = boxes[:, 2] * scale_x
            boxes[:, 3] = boxes[:, 3] * scale_y

            # Clip boxes to be within image dimensions (0 to image_size)
            boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], 0, self.image_size -1) # xmin, xmax
            boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], 0, self.image_size -1) # ymin, ymax
            
            # Filter out tiny boxes that might have become invalid after resize
            # e.g., width or height is less than 1 pixel
            valid_boxes_idx = (boxes[:, 2] - boxes[:, 0] >= 1) & (boxes[:, 3] - boxes[:, 1] >= 1)
            boxes = boxes[valid_boxes_idx]
            target['labels'] = target['labels'][valid_boxes_idx]
            target['difficulties'] = target['difficulties'][valid_boxes_idx]

        target['boxes'] = boxes

        # Convert to tensor
        img_tensor = T.ToTensor()(img_resized)
        
        # Normalize
        img_tensor = self.normalize(img_tensor)
        
        # For training, you might add data augmentation like random flips, color jitter, etc.
        # These would also need to adjust bounding boxes accordingly.
        # Example: Random horizontal flip
        if self.train and torch.rand(1) < 0.5 and boxes.numel() > 0:
            img_tensor = T.functional.hflip(img_tensor)
            # Flip boxes: x_min_new = image_width - x_max_old
            #             x_max_new = image_width - x_min_old
            new_xmin = self.image_size - boxes[:, 2]
            new_xmax = self.image_size - boxes[:, 0]
            boxes[:, 0] = new_xmin
            boxes[:, 2] = new_xmax
            target['boxes'] = boxes

        return img_tensor, target

# Collate function for DataLoader
# Needed because images in a batch can have different numbers of objects (and thus, boxes/labels)
def collate_fn(batch):
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    images = torch.stack(images, dim=0)
    return images, targets


# Test the dataset and dataloader
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src.config import IMAGE_SIZE, BATCH_SIZE, DATA_DIR, DEVICE, IDX_TO_CLASS

    # Create transforms
    train_transform = SSDTransform(image_size=IMAGE_SIZE, train=True)
    val_transform = SSDTransform(image_size=IMAGE_SIZE, train=False)

    # Create datasets
    # Using 'train' for both for quick testing, VOC2012 typically uses 'trainval' for training and 'test' for testing
    # Let's use VOC2007 'trainval' for training and 'test' for validation/testing, as it's often done.
    # Or stick to VOC2012 'train' and 'val'. Let's use 2012.
    # VOC2012 'train' has 5717 images, 'val' has 5823 images.
    try:
        train_dataset = PascalVOCDataset(root=DATA_DIR, year='2012', image_set='train', transforms=train_transform)
        val_dataset = PascalVOCDataset(root=DATA_DIR, year='2012', image_set='val', transforms=val_transform)
    except Exception as e:
        print(f"Error initializing dataset. Make sure {DATA_DIR} is writable and has network access.")
        print(f"If using VOC2012 'train' and 'val', ensure they exist. Torchvision downloads to root/VOCdevkit/VOC2012/")
        print(f"Error: {e}")
        exit()


    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("Training dataset is empty. Please check dataset path and download.")
        exit()

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Fetch a batch to test
    try:
        images, targets = next(iter(train_loader))
    except StopIteration:
        print("DataLoader is empty. This likely means the dataset was empty or an error occurred.")
        exit()
        
    print(f"\nBatch of images shape: {images.shape}") # Expected: (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    print(f"Length of targets list: {len(targets)}")   # Expected: BATCH_SIZE

    # Inspect first target in the batch
    if targets:
        print(f"First target in batch: (type: {type(targets[0])})")
        if isinstance(targets[0], dict):
            print(f"  Boxes shape: {targets[0]['boxes'].shape}")
            print(f"  Labels shape: {targets[0]['labels'].shape}")
            print(f"  Example labels: {targets[0]['labels']}")
        else:
            print(f"  Target content: {targets[0]}")


    # Visualize an image with its bounding boxes (optional)
    def visualize_image_with_boxes(img_tensor, target, idx_to_class_map, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # Denormalize
        img_tensor = img_tensor.clone()
        for t, m, s in zip(img_tensor, mean, std):
            t.mul_(s).add_(m)
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        img_pil = T.ToPILImage()(img_tensor)
        draw = ImageDraw.Draw(img_pil)
        
        boxes = target['boxes']
        labels = target['labels']
        
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        for i in range(boxes.shape[0]):
            box = boxes[i].tolist()
            label_idx = labels[i].item()
            class_name = idx_to_class_map.get(label_idx, "Unknown")
            
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1] - 15 if box[1] - 15 > 0 else box[1] + 1), class_name, fill="red", font=font)
        
        img_pil.show()

    # Visualize the first image of the batch
    if images.numel() > 0 and targets and 'boxes' in targets[0] and targets[0]['boxes'].numel() > 0:
        print("\nVisualizing first image of the batch (if matplotlib backend is available)...")
        visualize_image_with_boxes(images[0].cpu(), targets[0], IDX_TO_CLASS)
    else:
        print("\nSkipping visualization as there are no images or boxes in the first sample of the batch.")