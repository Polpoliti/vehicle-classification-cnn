
# 🚘 Fine-Grained Vehicle Classification from Scratch (196 Classes)

Hi,
As part of my personal deep learning journey, I wanted to challenge myself by solving a real-world, fine-grained image classification task — using **only a custom-built CNN from scratch**, without any pretrained weights or transfer learning.

I used the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), which includes over 16,000 images across 196 car categories. Some of the differences between classes are incredibly subtle (like small changes in headlights or grilles), which makes this problem both technically challenging and interesting.

---

## 🧠 What I Set Out to Do

- Build an image classification model **from the ground up**
- Handle **196 vehicle classes** with minimal visual differences
- Deal with class imbalance and overfitting effectively
- Learn and implement best practices in training, regularization, and model explainability

I intentionally avoided using transfer learning because I wanted to fully understand what it takes to build and train a deep learning model from scratch — architecturally, mathematically, and practically.

---

## 🏗️ What I Built

### Model Architecture

After several iterations, I developed a custom **ResNet-inspired CNN** architecture featuring:
- Residual blocks with skip connections
- Batch Normalization and ReLU activations
- Dropout before the final classification layer
- Global average pooling followed by a fully connected layer

> Simpler CNN architectures struggled to scale to 196 classes. Residual connections significantly improved convergence and generalization.


---

### Training Strategy

To make the most of the data and avoid overfitting, I combined several techniques:
- **MixUp augmentation** to help the model generalize better
- **WeightedRandomSampler** to address class imbalance
- **CrossEntropyLoss** with **label smoothing**
- **Strong data augmentations:**  including rotation, horizontal flips, brightness, random erasing, etc.
- **OneCycleLR learning rate scheduling with cosine annealing** for smoother convergence
- **EarlyStopping** to capture the best checkpoint before overfitting

> I originally tried simple oversampling and class weights, but combining the sampler with label smoothing gave significantly smoother training dynamics.

---

## 📊 Final Results

All results are from a custom CNN trained completely from scratch (no pretrained weights):

- ✅ **Best Validation Accuracy**: 70.9%
- 📉 **Best Validation Loss**: 1.5696
- 📊 **Train Accuracy at Best Epoch**: ~70.8%
- 🏁 **Epoch Reached**: **69** (early stopping triggered at epoch 69, best model saved at epoch 64)

Given the difficulty of the task (fine-grained, high-class-count, no transfer learning), I'm really happy with these results — and I see this as a strong baseline to build on further.

---

## 🔍 Model Explainability

To understand how the model makes predictions, I used **Grad-CAM** to visualize attention over image regions.  
It was great to see that the model consistently focused on relevant parts of the cars like front grilles, headlights, or brand-specific details — indicating that it was learning meaningful patterns and not just memorizing.

---

## 🧪 How to Run

To reproduce the results or train the model yourself, follow these steps:

### 1. 📦 Prepare the Dataset

Download and extract the dataset so the folder structure looks like this:



/Stanford_Cars_dataset-main
├── train/
├── test/


- The `train/` and `test/` folders should contain the raw car images, organized into subfolders by class (as expected by PyTorch's `ImageFolder`).
- You do **not** need annotation files like `.mat` or `label.labels.txt` — the model is trained directly from the image folder structure.
- No need for TFRecord — data is loaded efficiently using PyTorch's built-in tools.


### 2. 🧬 Clone the Repository

```bash
git clone https://github.com/Polpoliti/vehicle-classification-cnn.git
cd vehicle-classification-cnn
```

---

### 3. ⚙️ Install Dependencies

```bash
pip install -r requirements.txt
```

(Consider using a virtual environment.)

---

### 4. 📓 Run the Notebook

Launch Jupyter Notebook or your IDE, then open and run:

```
Final - ComputerVision_E2E_CNN_Cars196.ipynb
```

The notebook includes:
- Data parsing from `.mat` and `.txt` files
- Custom CNN architecture with residual blocks (no pretrained weights)
- Data augmentation, balancing, and training logic
- Accuracy tracking and Grad-CAM visualizations

---

### ✅ Tips

- A GPU is highly recommended for training (e.g., Colab Pro, or local CUDA setup)
- You can tweak hyperparameters directly in the notebook (batch size, LR, MixUp, etc.)
- At the end, you’ll see visual Grad-CAM outputs to better understand model decisions

---

## 📚 What I Learned

This project gave me deep, hands-on experience in:

- Building and debugging CNN architectures
- Tackling class imbalance with principled methods
- Designing data augmentation strategies
- Monitoring training behavior and using Grad-CAM for explainability
- Balancing performance with generalization — not just chasing accuracy

I gained a much stronger intuition around what works (and doesn’t) in deep learning, especially when building everything myself.
