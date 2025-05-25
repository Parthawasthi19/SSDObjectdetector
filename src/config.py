import torch
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 20 + 1  # 20 Pascal VOC classes + 1 background
IMAGE_SIZE = 300  # Input image size for SSD300

# Pascal VOC class names (ensure order matches dataset loading)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
# Create a mapping from class name to index (0 to 19 for objects, 20 could be background implicitly or explicitly handled)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(VOC_CLASSES)}


# Training parameters
BATCH_SIZE = 4 # Trainings
VAL_BATCH_SIZE = 1 # For validation
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 1 # Start with a smaller number for testing, e.g., 10
MOMENTUM = 0.9

# Anchor box settings (example, will be refined)
# These are aspect ratios for anchor boxes at different feature map scales
# For SSD300, typically 6 feature maps are used.
# We will define these more concretely when building the SSD head.
# For ResNet50, we might extract from fewer layers initially for simplicity.
# Let's define the layers from which we'll extract features and the number of anchor boxes per location.
# For SSD, common layers are Conv4_3, Conv7 (fc7), Conv8_2, Conv9_2, Conv10_2, Conv11_2 (using VGG terminology)
# For ResNet50, we'll tap into 'layer2', 'layer3', 'layer4' and add extra layers.
# Or, more simply for this example, let's just use the output of layer3 and layer4 from ResNet50
# and add two more custom convolutional layers to get 4 feature maps for detection.

# Feature maps we'll use and their properties (num_boxes per location)
# This will be determined by the SSD head architecture.
# For now, let's placeholder it, we'll define it properly in model.py
SSD_FEATURE_MAPS_INFO = [
    # {'name': 'from_resnet_layer2', 'num_boxes': 4}, # Example: (batch, 512, 38, 38)
    {'name': 'from_resnet_layer3', 'num_boxes': 6}, # Example: (batch, 1024, 19, 19)
    {'name': 'from_resnet_layer4', 'num_boxes': 6}, # Example: (batch, 2048, 10, 10)
    {'name': 'extra_conv1',        'num_boxes': 6}, # Example: (batch, 512, 5, 5)
    {'name': 'extra_conv2',        'num_boxes': 4}, # Example: (batch, 256, 3, 3)
    # Potentially more layers for smaller objects
]

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "ssd_resnet50_voc.pth")