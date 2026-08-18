# Reproducible Environment Setup

This folder contains the minimum required dependencies to reproduce the Siamese Localization project for submission evaluation.

## Requirements

The provided `requirements.txt` is a cleaned version that specifically pins the external packages actively used in the codebase:
- `opencv-python`
- `matplotlib`
- `numpy`
- `Pillow`
- `scipy`
- `torch` & `torchvision`
- `tqdm`

This avoids the bloat and potential OS-level conflicts of a full `pip freeze` (which can contain hundreds of unrelated system packages like ROS dependencies), guaranteeing a stable and targeted cross-platform installation.

## Environment Setup Instructions

Follow these exact steps to recreate the isolated Python environment on an Ubuntu/Linux machine:

### 1. Create a Virtual Environment
```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment
```bash
source venv/bin/activate
```

### 3. Install Dependencies
Run the following command to install the required packages. The `--extra-index-url` ensures you download the GPU-compatible PyTorch wheels for CUDA 12.1, matching the exact configuration used during development.
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Once installed, the project scripts can be run normally using this active virtual environment.
