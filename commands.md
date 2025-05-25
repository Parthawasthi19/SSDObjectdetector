# 🛠️ Project Setup – object_detection_project

Follow the steps below to set up and run the object detection project.

---

## 📁 1. Create and Activate Virtual Environment

Open your terminal in the root directory (`object_detection_project/`) and run:

```bash
# Create a virtual environment
python -m venv object_detection_env

# Activate the virtual environment

# On Windows (Git Bash or PowerShell)
source object_detection_env/Scripts/activate

# On macOS/Linux
source object_detection_env/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install PyTorch with appropriate CUDA version (adjust if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional libraries
pip install matplotlib numpy tqdm Pillow scikit-learn pycocotools torchmetrics

# Create a directory to save trained models
mkdir models

# Run utility functions
python -m src.utils

# Run dataset setup
python -m src.dataset

# Run model definitions
python -m src.model

# Run loss function module
python -m src.loss

python train.py

python predict.py

