# 🧬 Explainable AI-Based White Blood Cell (WBC) Classification Using EfficientNetV2L  

## 📘 Project Overview  
This project presents an **Explainable AI (XAI)**-driven approach for **automated classification of white blood cells (WBCs)** using the **Raabin-WBC dataset**.  
The proposed model leverages a **fine-tuned EfficientNetV2L** architecture for high-accuracy classification across five major WBC types — **basophils, eosinophils, lymphocytes, monocytes, and neutrophils** — while addressing dataset imbalance and enhancing model interpretability with **Grad-CAM** visualizations.  

---

## 📊 Dataset Description  
- **Dataset:** [Raabin-WBC Dataset](https://raabindata.com/)  
- **Microscopy setup:** Zeiss & Olympus CX18 (100× magnification)  
- **Staining method:** Giemsa stain  
- **Image acquisition:** Captured with Samsung Galaxy S5 and LG G3 smartphones  
- **Annotation:** Expert-labeled by two hematologists  
- **WBC classes:**  
  - Basophil  
  - Eosinophil  
  - Lymphocyte  
  - Monocyte  
  - Neutrophil  

The dataset exhibits **class imbalance**, with neutrophils dominating. To counter this, **data augmentation** and **label smoothing** were applied.

---

## 🧠 Model Architecture  

### 🔹 Backbone  
- **Base model:** EfficientNetV2L (pre-trained on ImageNet)  
- **Input size:** 224×224×3  
- **Transfer learning approach:**  
  - **Phase 1:** Train classifier head with frozen base  
  - **Phase 2:** Fine-tune last 50 layers with reduced learning rate  

### 🔹 Classification Head  
```python
Dense(128, activation='relu', kernel_regularizer=l2(1e-4))
Dropout(0.45)
Dense(256, activation='relu', kernel_regularizer=l2(1e-4))
Dropout(0.45)
Dense(5, activation='softmax')
```

### 🔹 Optimizer & Hyperparameters  
| Hyperparameter | Value | Description |
|----------------|--------|-------------|
| Optimizer | Adam | Adaptive learning optimizer |
| Learning rate | 1e-4 → 1e-5 | Reduced during fine-tuning |
| Batch size | Variable | Adjusted based on GPU memory |
| Loss | Categorical Crossentropy (label smoothing = 0.1) | Handles class noise & imbalance |
| Regularization | L2 (1e-4) | Prevents overfitting |
| Dropout | 0.45 | Randomly disables neurons for generalization |

---

## ⚙️ Training Strategy  

### 🧩 **Phase 1 – Feature Extraction**
- Freeze EfficientNetV2L layers  
- Train dense layers with Adam optimizer (`lr=1e-4`)  
- Use early stopping and checkpoint callbacks  

### 🔧 **Phase 2 – Fine-Tuning**
- Unfreeze last 50 layers of backbone  
- Retrain with a smaller learning rate (`1e-5`)  
- Use learning rate scheduler (`ReduceLROnPlateau`)  

---

## 📈 Performance Summary  

| Metric | Without Fine-tuning | With Fine-tuning |
|--------|----------------------|------------------|
| **Accuracy** | 88% | **99%** |
| **Macro Recall** | 69% | **97%** |
| **Macro F1-score** | 73% | **98%** |
| **Eosinophil Recall** | 21% → **97%** |
| **Monocyte Recall** | 49% → **93%** |

Fine-tuning led to significant improvement across minority classes, minimizing misclassifications and ensuring balanced detection across all WBC types.  

---

## 🔍 Explainable AI Integration  

To enhance interpretability and clinical trust, **XAI techniques** were applied:  
- **Grad-CAM:** Visualizes key regions influencing predictions.  
- **LIME (Local Interpretable Model-agnostic Explanations):** Provides pixel-level interpretability for individual predictions.  

### 📌 Recommended Slide Placement (for presentation)
| Visualization | Suggested Slide |
|----------------|-----------------|
| Confusion Matrix | After Results Table |
| ROC Curve | After Model Comparison |
| Grad-CAM Heatmaps | At the end of Results section |

---

## 🧪 Experimental Setup  
| Parameter | Details |
|------------|----------|
| Framework | TensorFlow 2.x / Keras |
| Hardware | NVIDIA GPU (CUDA enabled) |
| Image Size | 224 × 224 |
| Epochs | 15 (head training) + 30 (fine-tuning) |
| Augmentations | Rotation, flip, zoom, brightness/contrast adjustment, Gaussian blur |

---

## 📂 Repository Structure  

```
WBC-Classification-Using-EfficientNetV2L/
│
├── wbc-classification-using-efficientnetv2-l.ipynb      # Main notebook
├── README.md                     # Project documentation
├── requirements.txt               # Dependencies
├── /results/                     # Confusion matrix, ROC, Grad-CAM outputs
├── /models/                      # Trained model weights
└── /data/                        # Dataset or links to source
```

---

## 🚀 How to Run  

```bash
# Clone the repo
git clone https://github.com/pramod2594/WBC-Classification-Using-EfficientNetV2L.git
cd WBC-Classification-Using-EfficientNetV2L

# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook wbc-classification-using-efficientnetv2-l.ipynb
```
