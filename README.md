# Road Anomaly Detection

Our Road Anomaly Detection project. I've been working on using computer vision, specifically YOLOv11s models, to automatically spot issues like cracks and potholes on road surfaces. This repository contains the dataset details, the models I trained and used, evaluation results and the demo applications I built.

## Project Structure

```tree
Project Root
│
├── RAD_DATASET/
|
├── runs/                               # Training outputs
│   └── detect/
│       └── yolov11s_trained/
│           └── weights/
│               ├── best.pt             # Best trained model weights
|               └── best_saved_model    # TFLite model weights
│
├── test_output/                    # Inference output results
│
├── train.ipynb                     # Model training + evaluation notebook
├── inference.py                    # Image / Video / Live detection script
├── view_annotations.py             # YOLO annotation visualizer
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── yolov11n.pt                     # Pretrained YOLOv11 Nano model
├── yolov11s.pt                     # Pretrained YOLOv11 Small model

```

## Dataset

I put together a custom dataset specifically for training our main detection model.
The dataset structure within this repository looks like this:

```tree
RAD_DATASET/
├── images/
│   ├── train/
│   ├── valid/
|   └── test/
├── labels/
    ├── train/
    ├── valid/
    └── test/
```

## Model : Custom Trained YOLOv11s (`best.pt`)

This is the primary model I trained from scratch using our custom dataset.

- **Model Architecture:** YOLOv11s
- **Training Epochs:** 100
- **Training Time:** Approx. 30 hours
- **Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (6GB)
- **Best Weights File:** `/runs/detect/yolov11s_trained/weights/best.pt (Size: 18.2 MB)

### Validation Performance (`best.pt` during training)

These metrics reflect the performance on the validation set using the best weights saved during the training process.

| Class         | Precision | Recall    | mAP@.5    | mAP@.5:.95 |
| :------------ | :-------- | :-------- | :-------- | :--------- |
| **Overall**   | **0.753** | **0.716** | **0.753** | **0.438**  |
| Heavy-Vehicle | 0.921     | 0.976     | 0.979     | 0.764      |
| Light-Vehicle | 0.874     | 0.892     | 0.964     | 0.689      |
| Pedestrian    | 0.838     | 0.903     | 0.910     | 0.561      |
| RoadDamages   | 0.691     | 0.393     | 0.487     | 0.203      |
| Speed-Bump    | 0.645     | 0.729     | 0.746     | 0.411      |
| UnsurfaceRoad | 0.569     | 0.684     | 0.617     | 0.366      |

### Test Set Performance (`best.pt` - Final Evaluation)

I ran a final evaluation on a dedicated test set using the `best.pt` model.

| Class         | Precision | Recall    | mAP@.5    | mAP@.5:.95 |
| :------------ | :-------- | :-------- | :-------- | :--------- |
| **Overall**   | **0.743** | **0.726** | **0.763** | **0.441**  |
| Heavy-Vehicle | 0.921     | 0.976     | 0.979     | 0.764      |
| Light-Vehicle | 0.874     | 0.892     | 0.964     | 0.689      |
| Pedestrian    | 0.838     | 0.903     | 0.910     | 0.561      |
| RoadDamages   | 0.691     | 0.393     | 0.487     | 0.203      |
| Speed-Bump    | 0.645     | 0.729     | 0.746     | 0.411      |
| UnsurfaceRoad | 0.569     | 0.684     | 0.617     | 0.366      |

- **Average Inference Speed:** ~0.2 ms per image

#### Overall Test Metrics Summary:

- **Precision:** 0.743
- **Recall:** 0.726
- **mAP@0.5:** 0.763
- **mAP@0.5:0.95:** 0.441

---

## Deployment Guide – Raspberry Pi 4 Setup

### Step 0: Login Raspberry Pi

```bash
ssh username@hostname
```

### Step 1: Get Raspberry Pi IP Address

```bash
ifconfig
```

Copy the `inet` address under `wlan0`.

### Step 2: Enable VNC

```bash
sudo raspi-config
```

Navigate to:

- **Interface Options** → **VNC** → **Enable**

### Step 3: Connect via RealVNC

Use the copied `inet` IP address to log in using **RealVNC Viewer**.

### Step 4.1: Keyboard Layout

- **Pi Logo** → **Preferences** → **Control Center** → **Keyboard** → **English (US)**

### Step 4.2: Localization

- **Pi Logo** → **Preferences** → **Control Center** → **Localization** → **English (US)**

### Step 4.3: Enable SPI (Required for LoRa)

- **Pi Logo** → **Preferences** → **Control Center** → **Interfaces** → **Enable SPI**

### Step 5: Reboot

```bash
sudo reboot
```

### Step 6. System Update

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 7: Install pyenv

```bash
curl https://pyenv.run | bash
```

### Step 8.1: Configure pyenv

Add the following to `~/.bashrc`:

```bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc
```

### Step 8.2: Install Python Build Dependencies

```bash
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev tk-dev libffi-dev liblzma-dev
```

### Step 8.3: Install Python 3.11.8

```bash
pyenv install 3.11.8
```

### Step 9: Create Workspace

```bash
mkdir YOLO
cd YOLO
```

### Step 10: Set Local Python Version

```bash
pyenv local 3.11.8
```

### Step 11: Create Virtual Environment

```bash
python --version   # Should show Python 3.11.8
python -m venv --system-site-packages venv
source venv/bin/activate
```

### Step 12: Clone Repository

```bash
git clone https://github.com/imshubhamcodex/Real-Time-Road-Anomaly-Detection.git
cd Real-Time-Road-Anomaly-Detection/
```

### Step 13.1: Install packages

```bash
pip install -r requirements.txt
```

### Step 13.2: Export YOLO Model to NCNN or TFLite

```bash
yolo export model="./runs/detect/yolov11s_trained/weights/best.pt" format=ncnn imgsz=768
or
yolo export model=best.pt format=tflite int8=True data=data.yaml
```

> **Note:** `imgsz=768` must match `INFERENCE_SIZE` in the code.

### Step 14.1: Terminal-Only Mode (No Camera Preview)

Comment out the following line in `inference.py`:

```python
cv2.imshow("Live Feed", frame)
```

### Step 14.2: Camera Preview Mode

- Connect via **RealVNC**
- Activate the virtual environment

### Step 15: Run the Program

```bash
python inference.py
```

---

## Notes

- VNC is required for OpenCV GUI preview.
- NCNN, TFLite, ONNX export significantly improves inference speed on Raspberry Pi.

---

## Author

**Shubham Kumar**

Thanks
