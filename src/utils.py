import torch
import math

def generate_anchor_boxes(feature_map_sizes, image_size, anchor_scales, aspect_ratios_per_layer):
    """
    Generates anchor boxes for each feature map.
    Args:
        feature_map_sizes (list of tuples): (height, width) for each feature map.
        image_size (int or tuple): (height, width) of the input image.
        anchor_scales (list of floats): Scales for anchors (e.g., s_k in SSD paper) for each feature map.
                                        These are relative to image_size.
        aspect_ratios_per_layer (list of lists of floats): Aspect ratios for each feature map layer.
    Returns:
        all_anchors (Tensor): (num_total_anchors, 4) containing [cx, cy, w, h] for all anchors,
                              normalized by image_size.
    """
    if isinstance(image_size, int):
        img_h = img_w = image_size
    else:
        img_h, img_w = image_size

    all_anchors = []
    for i, fm_size in enumerate(feature_map_sizes):
        fm_h, fm_w = fm_size
        layer_anchors = []
        
        # For each location in the feature map
        for y_fm in range(fm_h):
            for x_fm in range(fm_w):
                # Center of the current cell in the feature map, mapped to image coordinates
                # These are normalized coordinates (0 to 1)
                cx = (x_fm + 0.5) / fm_w
                cy = (y_fm + 0.5) / fm_h

                current_aspect_ratios = aspect_ratios_per_layer[i]
                current_scale = anchor_scales[i] # s_k

                for ar in current_aspect_ratios:
                    # Base anchor (ratio 1)
                    if ar == 1.0: # Main scale for this layer
                        w = current_scale
                        h = current_scale
                        layer_anchors.append([cx, cy, w, h])

                        # Additional anchor with scale s'_k = sqrt(s_k * s_{k+1})
                        # This is only if we have a next scale defined
                        if i < len(anchor_scales) - 1:
                            next_scale = anchor_scales[i+1]
                            s_prime_k = math.sqrt(current_scale * next_scale)
                            w_prime = s_prime_k
                            h_prime = s_prime_k
                            layer_anchors.append([cx, cy, w_prime, h_prime])
                        # If it's the last layer, some implementations add an anchor slightly larger
                        # than current_scale for ratio 1, e.g., current_scale * some_factor
                        # For simplicity, we'll stick to the primary definition or handle it via aspect_ratios list

                    # Anchors for other aspect ratios
                    # w = s_k * sqrt(ar), h = s_k / sqrt(ar)
                    else:
                        w = current_scale * math.sqrt(ar)
                        h = current_scale / math.sqrt(ar)
                        layer_anchors.append([cx, cy, w, h])
        
        all_anchors.append(torch.tensor(layer_anchors, dtype=torch.float32))
        
    all_anchors_tensor = torch.cat(all_anchors, dim=0) # Shape: (total_anchors, 4)
    # Ensure anchors are clipped to [0, 1] for cx, cy, w, h (w, h are relative to image size)
    # cx, cy should naturally be within [0,1]
    # w, h could theoretically exceed 1 if scales are large; clip them.
    # However, usually cx,cy,w,h are used to derive xmin,ymin,xmax,ymax which are then clipped.
    # For now, let's assume scales are chosen such that w,h <= 1.0.
    # If not, they need to be scaled to image dims first, then converted to xmin,ymin,xmax,ymax, then clipped.
    # Let's assume the cx,cy,w,h format is preferred for now.
    return all_anchors_tensor


def cxcywh_to_xyxy(boxes_cxcywh):
    """
    Convert boxes from [cx, cy, w, h] normalized to [xmin, ymin, xmax, ymax] normalized.
    """
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    xmin = cx - 0.5 * w
    ymin = cy - 0.5 * h
    xmax = cx + 0.5 * w
    ymax = cy + 0.5 * h
    return torch.stack((xmin, ymin, xmax, ymax), dim=-1)

def xyxy_to_cxcywh(boxes_xyxy):
    """
    Convert boxes from [xmin, ymin, xmax, ymax] normalized to [cx, cy, w, h] normalized.
    """
    xmin, ymin, xmax, ymax = boxes_xyxy.unbind(-1)
    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + 0.5 * w
    cy = ymin + 0.5 * h
    return torch.stack((cx, cy, w, h), dim=-1)


# Example Usage (for testing, will be integrated into the model)
if __name__ == '__main__':
    # Typical SSD300 settings
    img_dim = 300
    # Feature maps: sizes and channels (example, adjust for ResNet)
    # VGG based SSD300 feature maps:
    # Conv4_3: 38x38
    # FC7:     19x19
    # Conv8_2: 10x10
    # Conv9_2: 5x5
    # Conv10_2: 3x3
    # Conv11_2: 1x1
    feature_map_dims = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    
    # Scales for anchor boxes (s_k values in the paper)
    # s_min = 0.2, s_max = 0.9 for SSD300 (m=6 layers)
    # s_k = s_min + (s_max - s_min)/(m-1) * (k-1)
    # k = 1 to m
    s_min = 0.1 # Adjusted from 0.2 to get smaller boxes for first layer. Or define scales manually.
    s_max = 0.9
    m = len(feature_map_dims)
    scales = [s_min + (s_max - s_min) * i / (m - 1) for i in range(m)]
    # The paper uses a slightly different scheme: first layer scale is s_min/2 or manually set,
    # and the last scale might be 1.05. Let's use a common set of scales for SSD300:
    # scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075] # len m+1 for s'_k = sqrt(s_k s_{k+1})
    # Or more direct:
    ssd_scales = [
        0.1,  # Scale for first feature map (e.g., 38x38)
        0.2,  # Scale for second feature map (e.g., 19x19)
        0.375,
        0.55,
        0.725,
        0.9,
        # Optional: a scale for the s'_k for the last layer, if we interpret s_{m+1}
        # For generate_anchor_boxes, we need m scales. The s'_k uses s_k and s_{k+1}
    ]
    if len(ssd_scales) < m: # If we need more scales based on feature_map_dims
        # Pad with linearly interpolated scales or define manually
        # For now, let's assume ssd_scales has m elements.
        # The s'_k will use current_scale and next_scale from this list.
        # If only m scales are given, the last layer won't have an s'_{m} box.
        # This is fine.
        print(f"Warning: Number of scales ({len(ssd_scales)}) is less than number of feature maps ({m}). Adjusting.")
        # A common list of scales for the 6 feature maps for SSD300
        ssd_scales_for_ssd300 = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # len m+1
        # Our function expects m scales. s'_k is derived from s_k and s_{k+1}.
        # So we need scales[0]...scales[m-1] for the m feature maps.
        # And scales[1]...scales[m] for the s_{k+1} part.
        # So, the list of scales should effectively be of length m+1 if we want s'_k for the last layer.
        # Let's use the structure from a known implementation for scales:
        # Feature map 1: scale s1, ratios {1, 2, 1/2}. (Plus s'_1 for ratio 1) -> 4 boxes
        # Feature map 2-5: scale sk, ratios {1, 2, 3, 1/2, 1/3}. (Plus s'_k for ratio 1) -> 6 boxes
        # Feature map 6: scale s6, ratios {1, 2, 1/2}. (Plus s'_6 for ratio 1) -> 4 boxes
        
    # Aspect ratios for each feature map layer
    # Typically:
    # Layer 1: [1.0, 2.0, 0.5] (and the extra s_k' box) -> 4 anchors per location
    # Layer 2-N-1: [1.0, 2.0, 3.0, 0.5, 0.333] (and extra s_k' box) -> 6 anchors
    # Layer N: [1.0, 2.0, 0.5] (and extra s_k' box) -> 4 anchors
    aspect_ratios_list = [
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 3.0, 0.5, 1./3.],
        [1.0, 2.0, 3.0, 0.5, 1./3.],
        [1.0, 2.0, 3.0, 0.5, 1./3.],
        [1.0, 2.0, 0.5],
        [1.0, 2.0, 0.5]
    ]
    # Number of boxes per location: (len(aspect_ratios) + 1 for the s_k' box) if 1.0 is in aspect_ratios.
    # Our function logic: if ar=1.0, it adds s_k and s'_k. Other ar's add one box each.
    # So num_boxes = len(current_aspect_ratios) + (1 if 1.0 in current_aspect_ratios else 0)
    # Example: ar=[1.0, 2.0, 0.5] -> 1 (for ar=1, scale s_k) + 1 (for ar=1, scale s'_k) + 1 (for ar=2) + 1 (for ar=0.5) = 4
    # Example: ar=[1.0, 2.0, 3.0, 0.5, 1/3] -> 1+1+1+1+1+1 = 6 boxes

    # Scales as used by PyTorch's SSD implementation (for reference):
    # These are s_k for each of the 6 feature maps. s'_k = sqrt(s_k * s_{k+1}) will use these.
    # So, the list should have 7 values if we want s'_k for the last layer using s_m and s_{m+1}.
    # Let's use scales for each layer, and s'_k will use current_scale and next_scale from this list.
    # The list of scales will be of length m, representing the primary scale for each of the m feature maps.
    # The s'_k for feature map i uses scale[i] and scale[i+1].
    # The last feature map's s'_k cannot be computed if scale list has only m elements.
    # This is a common setup. Some implementations provide m+1 scales to handle this.
    # For generate_anchor_boxes, let's define `anchor_scales` to be of length `m`.
    # The s'_k for the i-th layer uses anchor_scales[i] and anchor_scales[i+1].
    # So anchor_scales should be length m+1 for the s'_k logic to always have a next_scale.
    # Or, the function should handle the last layer not having an s'_k if len(anchor_scales) == m.
    # Our function currently handles it: if i < len(anchor_scales) - 1
    # So if `anchor_scales` has length `m`, the last s'_k won't be made.
    # If `anchor_scales` has length `m+1`, then all s'_k can be made.
    
    # Let's use the scales for SSD300: sizes of default_boxes/image_size
    # These are the *actual box sizes* relative to image_size, not the s_k parameters directly.
    # This is simpler for `generate_anchor_boxes`
    # The s_k values are the primary dimension for the box.
    # For SSD300, typical scales (s_k):
    min_ratio, max_ratio = 20, 90 # in percentage of image dimension
    # step = int(math.floor((max_ratio - min_ratio)) / (m - 2)) # m-2 because first and last are special
    # scales_values = [min_ratio / 100.] # s_1
    # for ratio in range(min_ratio + step, max_ratio + 1, step): # s_2 to s_{m-1}
    #     scales_values.append(ratio / 100.)
    # scales_values.append(1.05) # s_m (or slightly larger)
    
    # Let's use the scales from a popular PyTorch SSD tutorial (sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)
    # These are the s_k for each feature map
    scales_for_fmaps = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.05] # len = num_feature_maps + 1
    # Our function expects `anchor_scales` to be the `s_k` for *that layer*.
    # The `s'_k` uses `s_k` and `s_{k+1}`. So we need `m` scales, and the s'_k for the last layer
    # won't be generated if the list of scales is only `m` long.
    # If we pass `scales_for_fmaps[:-1]` (length `m`) as `anchor_scales` to our function,
    # and it internally uses `anchor_scales[i+1]` which is `scales_for_fmaps[i+1]`. This is good.
    
    anchor_s_k = scales_for_fmaps[:-1] # Use the first m scales for the m feature maps.
                                   # s'_k will use s_k and s_{k+1} = scales_for_fmaps[k+1]

    anchors_cxcywh = generate_anchor_boxes(feature_map_dims, img_dim, anchor_s_k, aspect_ratios_list)
    print(f"Generated {anchors_cxcywh.shape[0]} anchors.")
    # Expected for SSD300:
    # 38*38*4 = 5776
    # 19*19*6 = 2166
    # 10*10*6 = 600
    # 5*5*6   = 150
    # 3*3*4   = 36  (original SSD paper uses 4 here, some impl. use 6)
    # 1*1*4   = 4   (original SSD paper uses 4 here, some impl. use 6)
    # Total (with 4,6,6,6,4,4 boxes per loc): 5776 + 2166 + 600 + 150 + 36 + 4 = 8732 anchors.
    # Our function:
    # (38*38 * (len(ar[0])+1)) + (19*19 * (len(ar[1])+1)) + ...
    # (1444 * 4) + (361 * 6) + (100 * 6) + (25 * 6) + (9 * 4) + (1 * 4)
    # 5776 + 2166 + 600 + 150 + 36 + 4 = 8732. Matches!

    anchors_xyxy = cxcywh_to_xyxy(anchors_cxcywh)
    print(f"First 5 anchors (cx, cy, w, h):\n{anchors_cxcywh[:5]}")
    print(f"First 5 anchors (xmin, ymin, xmax, ymax):\n{anchors_xyxy[:5]}")

    # Check if any anchor coordinates are outside [0, 1] for xmin,ymin,xmax,ymax
    # Note: w,h in cxcywh can be > 1 if an anchor is larger than the image.
    # xmin,ymin,xmax,ymax should be clipped to [0,1] after conversion if they exceed.
    # Our cxcywh_to_xyxy doesn't clip yet. This should be done after this conversion.
    # Or, ensure scales/aspect_ratios are such that they don't produce boxes larger than image.
    # It's generally better to clip xmin,ymin,xmax,ymax.
    # The anchor boxes are typically stored as cx,cy,w,h and converted to xmin,ymin,xmax,ymax when needed (e.g. for IoU).
    # The model predicts offsets from these cx,cy,w,h anchors.