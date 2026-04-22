# 🧠 Brain Tumor Detection System

A production-ready end-to-end deep learning application that detects brain tumours from MRI scans using **VGG16 Transfer Learning** and visualises the model's attention using **Grad-CAM**.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Sample Output](#sample-output)
- [Tech Stack](#tech-stack)

---

## 🔬 Project Overview

This system uses a fine-tuned **VGG16** convolutional neural network (pretrained on ImageNet) to perform **binary classification** of brain MRI scans:

- **Tumor** — MRI shows signs of a brain tumour
- **No Tumor** — MRI appears normal

The model achieves high accuracy (typically **90%+** on the test set) and includes **Grad-CAM** visualisation to highlight which regions of the image influenced the prediction — making the model interpretable and trustworthy.

### Dataset

**Source**: [Kaggle – Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

| Split | Approx. size |
|-------|-------------|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

Dataset is auto-downloaded via `kagglehub` during training.

---

## 📁 Project Structure

```
brain-tumor-app/
│
├── data/                         # MRI images (auto-downloaded)
├── models/
│   ├── model.h5                  # Best model weights
│   ├── model_final.h5            # Final epoch model
│   ├── training_history.png      # Accuracy / Loss curves
│   ├── confusion_matrix.png      # Confusion matrix heatmap
│   ├── classification_report.txt # Full classification report
│   ├── performance_stats.json    # Metrics (used by Streamlit)
│   └── history.json              # Raw training history
│
├── notebooks/                    # Jupyter notebooks (EDA)
│
├── app/
│   └── streamlit_app.py          # ✅ Main Streamlit web app
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py            # Dataset download, augmentation, splits
│   ├── model_builder.py          # VGG16 model definition
│   ├── grad_cam.py               # Grad-CAM implementation
│   └── metrics.py                # Evaluation plots & reports
│
├── train.py                      # ✅ Full training pipeline
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python **3.9 – 3.11** (recommended: 3.10)
- `pip` (latest)
- (Optional) CUDA-capable GPU for faster training

### 1. Clone / Navigate to Project

```bash
cd brain-tumor-app
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle API Credentials

The training script auto-downloads the dataset using `kagglehub`. You need Kaggle API credentials:

1. Go to [https://www.kaggle.com](https://www.kaggle.com) → Account → Create API Token
2. This downloads `kaggle.json`
3. Place it at:
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
   - **macOS/Linux**: `~/.kaggle/kaggle.json`

---

## 🚀 How to Run

### Step 1: Train the Model

```bash
python train.py
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 30 | Maximum training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Initial learning rate |
| `--data-dir` | data | Data directory |
| `--model-dir` | models | Model output directory |
| `--patience` | 7 | EarlyStopping patience |

Example:
```bash
python train.py --epochs 50 --batch-size 16 --lr 5e-5
```

Training generates:
- `models/model.h5` — best model weights
- `models/training_history.png` — accuracy/loss curves
- `models/confusion_matrix.png` — confusion matrix
- `models/performance_stats.json` — metrics JSON

### Step 2: Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Open your browser at **[http://localhost:8501](http://localhost:8501)**

---

## 🏗️ Model Architecture

```
Input (224×224×3)
    ↓
VGG16 (ImageNet weights)
  [13 conv layers + 5 max-pool layers]
  [Layers 0–14: frozen | Layers 15+: fine-tuned]
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) → BatchNormalization → ReLU
    ↓
Dropout(0.5)               ← prevents overfitting
    ↓
Dense(1, Sigmoid)          ← binary output
    ↓
Output: P(Tumor) ∈ [0, 1]
```

**Training techniques:**
- Adam optimiser (lr = 1e-4)
- Binary cross-entropy loss
- EarlyStopping (patience = 7, monitors val_accuracy)
- ReduceLROnPlateau (factor = 0.5, patience = 3)
- Data augmentation: horizontal flip, rotation, zoom, shift
- Class-weighted loss (handles class imbalance)

---

## 📊 Sample Output

### Streamlit UI — Prediction Tab

| Feature | Description |
|---|---|
| Image upload | Drag & drop or click to browse |
| Prediction | "TUMOR DETECTED" 🔴 or "NO TUMOR DETECTED" 🟢 |
| Confidence | Probability score + animated bar |
| Grad-CAM | Side-by-side: original · heatmap · overlay |

### Streamlit UI — Performance Tab

| Feature | Description |
|---|---|
| Dataset overview | Total / train / val / test counts |
| Key metrics | Accuracy, loss, epochs |
| Classification report | Precision, Recall, F1 per class |
| Training curves | Accuracy & loss plots |
| Confusion matrix | Visual heatmap |

### Grad-CAM Explanation

The heatmap highlights the regions that most influenced the model's prediction. In tumor-positive scans, the activations typically concentrate around the **tumour mass**, providing a visual explanation of the model's reasoning.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Base Model | VGG16 (ImageNet) |
| Data Handling | NumPy, Pandas, scikit-learn |
| Image Processing | OpenCV, Pillow |
| Visualisation | Matplotlib |
| Web App | Streamlit |
| Dataset | kagglehub |

---

## ⚕️ Disclaimer

This system is built for **research and educational purposes only**. It is **not** a certified medical device and should **not** be used for clinical diagnosis. Always consult a qualified radiologist or physician for medical advice.

---

*Built with ❤️ using TensorFlow & Streamlit*
