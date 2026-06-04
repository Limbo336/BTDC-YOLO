# BTDC-YOLO: Lightweight Brain Tumor Detection and Classification in MRI

This repository contains the official implementation of **BTDC-YOLO**, a lightweight brain tumor detection and classification framework built upon YOLOv11## 🧰 

Requirements

- Python 3.11+
- PyTorch 2.7+
- CUDA 12.8 (recommended)
- Ultralytics 8.3.63

Usage

Clone the repo

Install requirements

Prepare your dataset (format: YOLO)

Train:

python train.py --data your_dataset.yaml --cfg your_model.yaml
# BTDC-YOLO: Lightweight Brain Tumor Detection and Classification in MRI

# 01. Overview of the Proposed Model
* This repository contains the official implementation of **BTDC-YOLO**, a lightweight brain tumor detection and classification framework built upon YOLOv11## 🧰 


# 02. Environment
- Python 3.11+
- PyTorch 2.7+
- CUDA 12.8 (recommended)
- Ultralytics 8.3.63
```
  pip install -r requirements.txt
```

# 03. Model Training Example
* python(start_train.py)
```
  python train.py --data your_dataset.yaml --cfg your_model.yaml
```
<div align=center>OR</div>

* Notebook(.ipynb)
  
```
from ultralytics import YOLO
from multiprocessing import freeze_support

# Set the model path
model_path = 'cfg your_model.yaml'

# Set the data configuration file path
data_path = 'data your_dataset.yaml'

# Load the model
model = YOLO(model_path)

if __name__ == '__main__':
    freeze_support()
    
     # Train the model
    model.train(data=data_path, epochs=100, project=project_path)
```

# 04. Model Validation Example
* Notebook(.ipynb)

```
from ultralytics import YOLO

# Load the model
model = YOLO("weights/best.pt")  # Load Custom Model

# Model Validation
metrics = model.val()  # No arguments needed, the dataset and settings are retained
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # List of mAP50-95 values for each category
```
