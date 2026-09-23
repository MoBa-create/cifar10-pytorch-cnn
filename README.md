# CIFAR-10 Image Classification with PyTorch & CNN

An end-to-end PyTorch implementation of Convolutional Neural Networks (CNN) for multi-class image classification on the CIFAR-10 dataset.

## 📊 Performance Benchmark

| Model | Architecture Highlights | Training Epochs | Test Accuracy |
| :--- | :--- | :---: | :---: |
| **SimpleCNN** | Baseline 2-Layer CNN | 10 | 69.34% |
| **ImprovedCNN** | Conv Blocks + BatchNorm + Dropout + CosineAnnealingLR | 15 | **82.28%** |

---

## 🛠️ Key Improvements in `ImprovedCNN`

1. **Data Augmentation:** Applied `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()` during training to increase visual variance and generalize well.
2. **Batch Normalization (`nn.BatchNorm2d`):** Stabilized internal covariate shift across intermediate layers for accelerated convergence.
3. **Regularization (`nn.Dropout`):** Dropped 20%-50% of feature activations to prevent over-fitting.
4. **Learning Rate Scheduler:** Integrated `CosineAnnealingLR` for adaptive learning rate decay across epochs.
5. **Robust Path Handling:** Standardized local filesystem relative/absolute paths via `os.path.abspath` for platform-agnostic execution.

---

## 💻 Environment & Hardware Specs

- **Framework:** PyTorch & Torchvision
- **Language:** Python 3.13
- **Hardware Acceleration:** NVIDIA GeForce RTX 5070 Ti Laptop GPU (CUDA)

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install required packages:
```bash
cd cifar10-cnn-classifier
pip install -r requirements.txt

2. Model Training

Train the improved CNN model:

python train_improved_cifar10.py


3. Inference & Visualization

Run predictions on 8 random test samples and display visual output:


python predict_random.py
