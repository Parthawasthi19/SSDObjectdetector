
---

**`README.md`**

```markdown
# Object Detection with Custom SSD Head on ResNet50 Backbone

## Project Overview

This project implements an object detection model based on the Single Shot Detector (SSD) architecture. A custom SSD detection head is added to a pre-trained ResNet50 CNN backbone. The model is trained on the Pascal VOC dataset (configurable for VOC2007 or VOC2012) and evaluated using Mean Average Precision (mAP).

This project serves as a hands-on exercise in understanding and building computer vision architectures, particularly for object detection, and explores the integration of pre-trained models with custom detection layers.

## Assignment Objectives Met

1.  **Select an appropriate CNN backbone architecture:** ResNet50 was chosen for its balance of performance and availability of pre-trained weights.
2.  **Implement object detection layers on top of the chosen backbone:** An SSD-style multi-scale detection head was implemented, including auxiliary convolutional layers for generating feature maps at various resolutions and prediction layers for bounding box regression and classification.
3.  **Train the model on an open-source dataset:** The model is configured to train on the Pascal VOC dataset (VOC2007 `trainval` and `test` sets by default, configurable to VOC2012).
4.  **Evaluate model performance:** The model is evaluated using validation loss and Mean Average Precision (mAP) via the `torchmetrics` library.
5.  **Document your experience and learnings:** This README and the development process serve as documentation.

## Features

*   **Backbone:** Pre-trained ResNet50 from `torchvision`.
*   **Detection Head:** SSD-style multi-scale head with:
    *   Feature maps extracted from ResNet50 (`layer2`, `layer3`, `layer4`).
    *   Additional auxiliary convolutional layers to create finer-grained feature maps.
    *   Prediction convolutions for each selected feature map to output localization offsets and class confidences.
*   **Anchor Boxes:** Generated programmatically based on feature map scales and predefined aspect ratios.
*   **Loss Function:** Custom MultiBox Loss, combining:
    *   Smooth L1 Loss for localization (bounding box regression).
    *   Cross-Entropy Loss for classification.
    *   Hard Negative Mining to handle class imbalance.
*   **Dataset:** Pascal VOC (2007 or 2012), with data loading, parsing, and transformations (including bounding box adjustments).
*   **Training:**
    *   SGD optimizer with momentum and weight decay.
    *   Learning rate scheduler (`ReduceLROnPlateau` based on mAP@0.50).
    *   Gradient clipping to stabilize training.
    *   Model checkpointing for the best performing model on the validation set.
*   **Evaluation:**
    *   Validation loss (localization and confidence).
    *   Mean Average Precision (mAP, mAP@0.50, mAP@0.75) using `torchmetrics`. Logic to offload metric calculations to CPU to conserve GPU memory.
*   **Inference:** Script to load a trained model and predict objects on new images, visualizing the results.

## Project Structure

## Setup and Installation

1.  **Clone the repository (if applicable) or ensure all files are in the structure above.**
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv object_detection_env
    # Activate on Windows:
    source object_detection_env/Scripts/activate
    # Activate on macOS/Linux:
    source object_detection_env/bin/activate
    ```
3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    torch>=1.10.0 # Or your specific version, e.g., 2.0.0
    torchvision>=0.11.0 # Or e.g., 0.15.0
    torchaudio>=0.10.0 # Or e.g., 2.0.0
    matplotlib
    numpy
    tqdm
    Pillow
    scikit-learn
    pycocotools # Often a dependency for mAP or certain datasets
    torchmetrics>=0.7.0 # Or a recent version
    ```
    Then install using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For PyTorch with CUDA support, it's often best to install it directly from the PyTorch website's instructions for your specific CUDA version, e.g.:*
    `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

4.  **Create `models` directory:**
    If it doesn't exist, create it in the project root: `mkdir models`

## How to Run

Please refer to `COMMANDS.md` for detailed step-by-step instructions on:
*   Testing individual components (`src/utils.py`, `src/dataset.py`, etc.).
*   Training the model (`train.py`).
*   Running inference on new images (`predict.py`).

### Key Configuration (`src/config.py`)

*   `DEVICE`: Automatically detects CUDA if available, else CPU.
*   `IMAGE_SIZE`: Input image dimensions for the model (default: 300).
*   `BATCH_SIZE`, `VAL_BATCH_SIZE`: Batch sizes for training and validation.
*   `LEARNING_RATE`, `NUM_EPOCHS`, etc.: Hyperparameters for training.
*   `DATA_DIR`, `MODEL_SAVE_PATH`: Paths for data and saved models.

## Implementation Details

### CNN Backbone (ResNet50)
A pre-trained ResNet50 is loaded from `torchvision.models`. The `create_feature_extractor` utility is used to extract feature maps from intermediate layers (`layer2`, `layer3`, `layer4`), which serve as the base for multi-scale detection.

### SSD Head
1.  **Auxiliary Convolutions:** A series of convolutional layers are appended to the last ResNet50 feature map (`layer4` output) to produce additional feature maps at progressively smaller spatial resolutions (e.g., 5x5, 3x3, 1x1 for a 300x300 input).
2.  **Prediction Convolutions:** For each selected feature map (from both the backbone and auxiliary layers), two parallel 3x3 convolutional layers predict:
    *   **Localization offsets:** 4 values per anchor box, refining its position and size.
    *   **Class confidences:** Scores for each class (including background) per anchor box.
3.  **Anchor Boxes:** Generated prior to training for each prediction layer. These default boxes have varying scales and aspect ratios, designed to cover objects of different sizes and shapes. The model learns to predict offsets from these anchors.

### Loss Function (MultiBoxLoss)
The loss is a weighted sum of localization loss and confidence loss:
*   **Matching:** Ground truth boxes are matched to anchor boxes based on Intersection over Union (IoU). An anchor is positive if its IoU with a ground truth box is above a threshold (e.g., 0.5), or if it's the best-matching anchor for a particular ground truth box.
*   **Localization Loss (Smooth L1):** Applied only to positive anchor matches. It penalizes errors in the predicted bounding box offsets.
*   **Confidence Loss (Cross-Entropy):** Applied to both positive matches and a selected set of negative matches (Hard Negative Mining). This prevents the model from being overwhelmed by the vast number of easy negative anchors.
*   **Hard Negative Mining:** Selects negative anchors with the highest confidence loss, typically maintaining a ratio (e.g., 3:1) of negatives to positives.

### Training and Evaluation
*   The model is trained end-to-end using an SGD optimizer.
*   During evaluation, predictions are decoded by applying offsets to anchors, followed by Non-Maximum Suppression (NMS) to eliminate redundant detections.
*   Mean Average Precision (mAP) is the primary metric for object detection performance.

## Potential Challenges & Learnings

*   **Memory Management:** Object detection models are memory-intensive. Careful management of batch sizes (especially `VAL_BATCH_SIZE`) and strategies like offloading metric computations to CPU are crucial for systems with limited VRAM.
*   **Numerical Stability:** NaN losses can occur due to issues like log(0), division by zero, or exploding gradients. Solutions include using appropriate epsilons, gradient clipping, careful learning rate selection, and consistent variance scaling in localization targets.
*   **Anchor Box Design:** The scales and aspect ratios of anchor boxes significantly impact the model's ability to detect objects of different sizes.
*   **Hyperparameter Tuning:** Finding optimal hyperparameters (learning rate, batch size, optimizer settings, loss weights) is an iterative process.
*   **Decoding and NMS:** The post-processing steps to convert raw model outputs into final detections are critical for good evaluation results and visual output.

## Future Improvements

*   Experiment with different backbone architectures (e.g., MobileNetV2/V3, EfficientNet).
*   Implement a Feature Pyramid Network (FPN) for potentially better feature fusion.
*   More extensive data augmentation.
*   Train on larger datasets like COCO.
*   Full hyperparameter optimization.
*   More sophisticated NMS techniques during evaluation.