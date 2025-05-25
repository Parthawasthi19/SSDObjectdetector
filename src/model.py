# src/model.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from src.config import NUM_CLASSES, DEVICE, IMAGE_SIZE # Ensure IMAGE_SIZE is imported
from src.utils import generate_anchor_boxes # Assuming utils.py is in the same directory

# --- Backbone ---
def get_resnet50_backbone(pretrained=True, requires_grad=True):
    """
    Loads a ResNet50 model and sets it up for feature extraction.
    """
    resnet50_weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    resnet50 = models.resnet50(weights=resnet50_weights)
    
    # Freeze parameters if requires_grad is False for the whole backbone
    if not requires_grad:
        for param in resnet50.parameters():
            param.requires_grad = False
    # Example for more granular control (e.g., fine-tune only layer3 and layer4):
    # else: # if requires_grad is True, but we want to freeze specific earlier layers
    #     for name, param in resnet50.named_parameters():
    #         if not ('layer3' in name or 'layer4' in name or 'fc' in name): # Keep fc trainable too if not removed
    #             param.requires_grad = False


    # Define nodes for feature extraction
    # For SSD300 on ResNet50, typical outputs:
    # layer2: ~38x38 output (after ResNet's layer2 block)
    # layer3: ~19x19 output (after ResNet's layer3 block)
    # layer4: ~10x10 output (after ResNet's layer4 block)
    return_nodes = {
        'layer2': 'feat_layer2', # (N, 512, H/8, W/8) -> e.g., 38x38 for 300x300 input
        'layer3': 'feat_layer3', # (N, 1024, H/16, W/16) -> e.g., 19x19 for 300x300 input
        'layer4': 'feat_layer4'  # (N, 2048, H/32, W/32) -> e.g., 10x10 for 300x300 input
    }
    
    # Create the feature extractor
    # Important: We are extracting from the original resnet50.
    # The requires_grad logic above controls what parts of this extractor will be trained.
    feature_extractor = create_feature_extractor(resnet50, return_nodes=return_nodes)
    return feature_extractor

# --- Auxiliary Convolutional Layers (for SSD multi-scale features) ---
class AuxiliaryConvolutions(nn.Module):
    def __init__(self, input_channels_layer4=2048):
        super().__init__()
        # These layers will take the output of ResNet's layer4 
        # (e.g., input_channels_layer4=2048 channels, 10x10 feature map for 300x300 input)
        # and produce smaller feature maps.

        # Extra Layer 1: 10x10 -> 5x5 (for 300x300 input)
        self.conv_extra1_1 = nn.Conv2d(input_channels_layer4, 256, kernel_size=1, padding=0)
        self.conv_extra1_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # Output: (N, 512, 5, 5)

        # Extra Layer 2: 5x5 -> 3x3
        self.conv_extra2_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv_extra2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # Output: (N, 256, 3, 3)
        
        # Extra Layer 3: 3x3 -> 1x1
        self.conv_extra3_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        # For 3x3 -> 1x1 with kernel_size=3:
        # if stride=1, padding must be 0. Output_size = (Input_size - Kernel_size + 2*Padding)/Stride + 1
        # (3 - 3 + 2*0)/1 + 1 = 1. Correct.
        # if stride=2, padding must be 0. (3-3+0)/2 + 1 = 1. Also correct. (Can use stride=2, padding=0 or stride=1,padding=0 for last layer if desired)
        # Original SSD papers sometimes use different kernel sizes for the 1x1 map (e.g., kernel 1x1 or 3x3 with padding 'valid')
        # Let's use stride 1, padding 0 for this last one to match common VGG-based SSDs which might do global pooling or 3x3 no pad.
        self.conv_extra3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0) # Output: (N, 256, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat_layer4):
        out = self.relu(self.conv_extra1_1(feat_layer4))
        feat_extra1 = self.relu(self.conv_extra1_2(out))

        out = self.relu(self.conv_extra2_1(feat_extra1))
        feat_extra2 = self.relu(self.conv_extra2_2(out))

        out = self.relu(self.conv_extra3_1(feat_extra2))
        feat_extra3 = self.relu(self.conv_extra3_2(out))
        
        return [feat_extra1, feat_extra2, feat_extra3]


# --- Prediction Convolutional Layers (for loc and conf) ---
class PredictionConvolutions(nn.Module):
    def __init__(self, num_classes, feature_map_channels_list, anchors_per_location_list):
        """
        Args:
            num_classes (int): Number of object classes (including background).
            feature_map_channels_list (list of int): Number of channels for each feature map input.
            anchors_per_location_list (list of int): Number of anchors for each feature map.
        """
        super().__init__()
        self.num_classes = num_classes
        self.loc_convs = nn.ModuleList()
        self.conf_convs = nn.ModuleList()

        if len(feature_map_channels_list) != len(anchors_per_location_list):
            raise ValueError("Length of feature_map_channels and anchors_per_location must match.")

        for i in range(len(feature_map_channels_list)):
            # Localization Prediction: 4 coordinates (dx, dy, dw, dh) for each anchor
            self.loc_convs.append(
                nn.Conv2d(feature_map_channels_list[i], anchors_per_location_list[i] * 4, kernel_size=3, padding=1)
            )
            # Classification Prediction: num_classes scores for each anchor
            self.conf_convs.append(
                nn.Conv2d(feature_map_channels_list[i], anchors_per_location_list[i] * self.num_classes, kernel_size=3, padding=1)
            )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, all_feature_maps_for_prediction):
        # all_feature_maps_for_prediction: list of features [feat_layer2, feat_layer3, feat_layer4, feat_extra1, ...]
        
        loc_preds_batch = []
        conf_preds_batch = []

        for i, feat_map in enumerate(all_feature_maps_for_prediction):
            # Localization predictions
            loc_pred = self.loc_convs[i](feat_map) # (N, num_anchors_at_loc*4, H, W)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous() # (N, H, W, num_anchors_at_loc*4)
            loc_pred = loc_pred.view(loc_pred.size(0), -1, 4) # (N, H*W*num_anchors_at_loc, 4)
            loc_preds_batch.append(loc_pred)

            # Confidence predictions
            conf_pred = self.conf_convs[i](feat_map) # (N, num_anchors_at_loc*num_classes, H, W)
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous() # (N, H, W, num_anchors_at_loc*num_classes)
            conf_pred = conf_pred.view(conf_pred.size(0), -1, self.num_classes) # (N, H*W*num_anchors_at_loc, num_classes)
            conf_preds_batch.append(conf_pred)
            
        final_loc_preds = torch.cat(loc_preds_batch, dim=1)     # (N, total_anchors, 4)
        final_conf_preds = torch.cat(conf_preds_batch, dim=1) # (N, total_anchors, num_classes)
        
        return final_loc_preds, final_conf_preds


# --- The Full SSD Model ---
class SSDResNet50(nn.Module):
    def __init__(self, num_classes, image_size=300, pretrained_backbone=True, backbone_requires_grad=True):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size # Assuming square images for simplicity in feature map size calculation

        self.backbone = get_resnet50_backbone(pretrained=pretrained_backbone, requires_grad=backbone_requires_grad)
        
        # Determine output channels from backbone layers to feed into auxiliary and prediction layers
        # These are typical for ResNet50:
        # layer2 out: 512 channels
        # layer3 out: 1024 channels
        # layer4 out: 2048 channels
        self.aux_convs = AuxiliaryConvolutions(input_channels_layer4=2048)

        # Define feature map properties (sizes, channels, aspect ratios for anchors)
        # Sizes are approximate for a 300x300 input. Exact sizes depend on padding/stride details in ResNet.
        # The sizes here are crucial for anchor generation.
        # ResNet50 for 300x300 input:
        # Stem (conv1, maxpool): 300 -> 150 -> 75 (stride 2, stride 2)
        # layer1: no spatial reduction, channels 256
        # layer2: first block stride 1 (spatial), output 75x75 (channels 512). If first block stride 2, then 38x38.
        # Standard ResNet: conv1 (s2), pool (s2) -> /4. layer1 (s1). layer2 (s2) -> /8. layer3 (s2) -> /16. layer4 (s2) -> /32.
        # So, 300/8 = 37.5 -> 38 (layer2)
        # 300/16 = 18.75 -> 19 (layer3)
        # 300/32 = 9.375 -> 10 (layer4)
        self.feature_map_definitions = [
            # From Backbone
            {'name': 'layer2',    'size': (38, 38), 'channels': 512,  'aspect_ratios': [1.0, 2.0, 0.5]},
            {'name': 'layer3',    'size': (19, 19), 'channels': 1024, 'aspect_ratios': [1.0, 2.0, 3.0, 0.5, 1./3.]},
            {'name': 'layer4',    'size': (10, 10), 'channels': 2048, 'aspect_ratios': [1.0, 2.0, 3.0, 0.5, 1./3.]},
            # From AuxiliaryConvolutions (outputs of conv_extraX_2 layers)
            {'name': 'aux_conv1', 'size': (5, 5),   'channels': 512,  'aspect_ratios': [1.0, 2.0, 3.0, 0.5, 1./3.]},
            {'name': 'aux_conv2', 'size': (3, 3),   'channels': 256,  'aspect_ratios': [1.0, 2.0, 0.5]},
            {'name': 'aux_conv3', 'size': (1, 1),   'channels': 256,  'aspect_ratios': [1.0, 2.0, 0.5]}
        ]
        
        fm_dims_for_anchors = [d['size'] for d in self.feature_map_definitions]
        # Scales (s_k) for each feature map. List of m+1 values for m feature maps.
        # s_k for i-th map is anchor_s_k_values[i], s_{k+1} is anchor_s_k_values[i+1]
        self.anchor_s_k_values = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # len=num_feature_maps + 1
        
        aspect_ratios_list_for_anchors = [d['aspect_ratios'] for d in self.feature_map_definitions]

        # Generate anchors using the correct keyword arguments
        self.anchors_cxcywh = generate_anchor_boxes(
            feature_map_sizes=fm_dims_for_anchors,
            image_size=self.image_size,
            anchor_scales=self.anchor_s_k_values, # Pass the list of m+1 scales
            aspect_ratios_per_layer=aspect_ratios_list_for_anchors
        )
        # Anchors should be on the same device as model parameters and input data during forward pass.
        # It's good practice to register them as buffers if they are part of the model state,
        # or ensure they are moved to device here or before use in loss/inference.
        # For now, move it here. If registered as buffer, .to(device) in main model works.
        self.anchors_cxcywh = self.anchors_cxcywh.to(DEVICE)


        # Calculate number of anchors per location for PredictionConvolutions
        # This is len(aspect_ratios) + 1 (for the extra s'_k box if 1.0 is an aspect ratio)
        # This logic must match how generate_anchor_boxes actually creates them.
        # generate_anchor_boxes: if ar=1.0, adds s_k box and s'_k box. Others add one box.
        # So num_boxes_at_loc = len(ars_for_layer) + (1 if 1.0 in ars_for_layer else 0)
        anchors_per_loc_list = []
        for ar_list in aspect_ratios_list_for_anchors:
            count = len(ar_list)
            if 1.0 in ar_list: # If 1.0 is present, an s_prime_k box is also added by generate_anchor_boxes
                count +=1
            anchors_per_loc_list.append(count)
        
        pred_conv_input_channels_list = [d['channels'] for d in self.feature_map_definitions]
        self.pred_convs = PredictionConvolutions(
            self.num_classes,
            pred_conv_input_channels_list,
            anchors_per_loc_list
        )


    def forward(self, image_batch):
        # 1. Get base feature maps from ResNet backbone
        # Output of create_feature_extractor is a dict: {'feat_layer2': tensor, ...}
        backbone_feats_dict = self.backbone(image_batch)
        
        # Extract features in the order defined for SSD
        feat_l2 = backbone_feats_dict['feat_layer2']
        feat_l3 = backbone_feats_dict['feat_layer3']
        feat_l4 = backbone_feats_dict['feat_layer4']
        
        # 2. Get auxiliary feature maps
        # aux_convs takes feat_l4 (e.g., 10x10 from ResNet layer4)
        # and produces [feat_extra1 (5x5), feat_extra2 (3x3), feat_extra3 (1x1)]
        aux_feats_list = self.aux_convs(feat_l4) # This returns a list of features

        # Combine all feature maps in the order expected by PredictionConvolutions
        # and matching the order of anchor generation.
        # Order: layer2, layer3, layer4, aux_conv1_out, aux_conv2_out, aux_conv3_out
        all_feature_maps = [feat_l2, feat_l3, feat_l4] + aux_feats_list
        
        # 3. Get localization and confidence predictions
        # loc_preds: (N, total_anchors, 4)
        # conf_preds: (N, total_anchors, num_classes)
        loc_preds, conf_preds = self.pred_convs(all_feature_maps)
        
        return loc_preds, conf_preds

# Test the full model
if __name__ == '__main__':
    # Ensure config variables are accessible or re-define them for the test
    # from src.config import IMAGE_SIZE, NUM_CLASSES, DEVICE (already imported at top)
    
    print("Initializing SSDResNet50 model...")
    # Use the imported IMAGE_SIZE from config
    model = SSDResNet50(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, backbone_requires_grad=False) # Set backbone_requires_grad=False for faster test
    model.to(DEVICE)
    model.eval() # Set to evaluation mode for testing

    print(f"Model image input size: {model.image_size}")
    print(f"Number of generated anchor boxes: {model.anchors_cxcywh.shape[0]}")
    # Expected: 8732 for the standard SSD300 configuration with (4,6,6,6,4,4) anchors per location

    # Create a dummy input tensor
    # Use model.image_size to ensure consistency
    dummy_input = torch.randn(2, 3, model.image_size, model.image_size).to(DEVICE) # Batch size 2
    print(f"Dummy input shape: {dummy_input.shape}")

    with torch.no_grad(): # Important for testing/inference
        loc_preds, conf_preds = model(dummy_input)

    print(f"Location predictions shape: {loc_preds.shape}")   # Expected: (2, num_total_anchors, 4)
    print(f"Confidence predictions shape: {conf_preds.shape}") # Expected: (2, num_total_anchors, num_classes)

    # Check if total anchors match prediction shapes
    num_total_anchors_generated = model.anchors_cxcywh.shape[0]
    assert loc_preds.shape[1] == num_total_anchors_generated, \
        f"Mismatch in number of anchors for loc_preds. Expected {num_total_anchors_generated}, got {loc_preds.shape[1]}"
    assert conf_preds.shape[1] == num_total_anchors_generated, \
        f"Mismatch in number of anchors for conf_preds. Expected {num_total_anchors_generated}, got {conf_preds.shape[1]}"
    
    print("Model forward pass test successful!")

    # Further test on backbone feature extraction (optional detailed check)
    print("\nTesting backbone feature extraction part:")
    backbone_test = get_resnet50_backbone(pretrained=False, requires_grad=False).to(DEVICE).eval()
    dummy_backbone_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    features = backbone_test(dummy_backbone_input)
    print("Backbone Output Features (for dummy input):")
    for name, tensor in features.items():
        print(f"{name}: {tensor.shape}")
    # Expected for 300x300 input:
    # feat_layer2: torch.Size([1, 512, 38, 38])
    # feat_layer3: torch.Size([1, 1024, 19, 19])
    # feat_layer4: torch.Size([1, 2048, 10, 10])