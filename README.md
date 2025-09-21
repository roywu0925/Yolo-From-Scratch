
##  Project Overview

This project is a **from-scratch reimplementation of YOLO (You Only Look Once)** for object detection. It aims to provide a hands-on understanding of the YOLO architecture, loss function, and training pipeline, instead of relying solely on pre-built frameworks. The project includes both **custom loss function implementation** and **training notebooks**.

Use cases: Education, research, and experimentation with deep learning models for object detection.

---

##  Repository Contents

* **`yolo_loss.py`** → Custom YOLO loss function implementation
* **`my_mp4.ipynb`** → Jupyter notebook for experimentation and training
* **`src/`** → Source code (model components, data pipeline, utils)
* **`data/`** → Dataset folder (not included; add your training/validation data here)
* **`download_data.sh`** → Script to download datasets
* **`README.md`** → Project documentation (this file)

---

##  Key Features

* **YOLO Architecture**: Implemented from scratch for transparency
* **Custom Loss Function**: `yolo_loss.py` implements bounding box regression, objectness, and classification terms
* **Training Notebook**: End-to-end workflow in `my_mp4.ipynb`
* **Extensible**: Easy to modify architecture or loss for research

---

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/roywu0925/Yolo-From-Scratch.git
cd Yolo-From-Scratch
```

### 2. Set Up Environment

Using pip:

```bash
pip install -r requirements.txt
```

Or with Poetry (if configured):

```bash
poetry install
```

---

## ⚡ Usage

### Training / Experimentation

Open the notebook:

```bash
jupyter notebook my_mp4.ipynb
```

Follow the steps to train or test the model.

### Loss Function Testing

You can test the YOLO loss function separately:

```bash
python yolo_loss.py
```

### Data

* Place your dataset in the `data/` folder
* Or use the provided script:

```bash
bash download_data.sh
```

---

##  Key Takeaways

1. Building YOLO from scratch helps to deeply understand object detection internals
2. Loss balancing (localization, confidence, classification) is critical
3. Codebase is structured for extensibility and research

---

##  References

* [YOLOv1 Paper](https://arxiv.org/abs/1506.02640)
* [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
* [YOLOv5/Ultralytics Implementation](https://github.com/ultralytics/yolov5)

---

##  Author

* **Roy Wu**
  Statistics + Computer Science + Math background
  Focus: AI systems, computer vision, and model reimplementation for learning
