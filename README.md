
# 🚘 Fine Grained Vehicle Classification from Scratch (196 Classes)

Hi,
As part of my personal deep learning journey, I wanted to challenge myself by solving a real world, fine grained image classification task - using **only a custom-built CNN from scratch**, without any pretrained weights or transfer learning.

I used the [Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), which includes over 16,000 images across 196 car categories. Some of the differences between classes are incredibly subtle (like small changes in headlights or grilles), which makes this problem both technically challenging and interesting.

---

## What I Set Out to Do

- Build an image classification model **from the ground up**
- Handle **196 vehicle classes** with minimal visual differences
- Deal with class imbalance and overfitting effectively
- Learn and implement best practices in training, regularization, and model explainability

I intentionally avoided using transfer learning because I wanted to fully understand what it takes to build and train a deep learning model from scratch - architecturally, mathematically, and practically.

---

## What I Built

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

## 📊 Final Performance Results

All results are from a custom CNN trained completely from scratch (no pretrained weights):

-  **Best Validation Accuracy**: 70.9%
-  **Best Validation Loss**: 1.5696
-  **Train Accuracy at Best Epoch**: ~70.8%
-  **Training Stopped at Epoch**: 69 (early stopping triggered at epoch 69, best model saved at epoch 64)

Considering the complexity of the task - **fine grained classification across 196 visually similar classes, with no external features or pretrained knowledge** - these results demonstrate strong model performance and robust generalization. This serves as a solid baseline for future enhancements.

---

##  Model Explainability

To understand how the model makes predictions, I used **GradCAM** to visualize attention over image regions.  
It was great to see that the model consistently focused on relevant parts of the cars like front grilles, headlights, or brand specific details - indicating that it was attending to discriminative features and not just relying on spurious cues.

---

## How to Run

To reproduce the results or train the model yourself, follow these steps:

### 1. Prepare the Dataset

Download and extract the dataset so the folder structure looks like this:



/Stanford_Cars_dataset-main
├── train/
├── test/


- The `train/` and `test/` folders should contain the raw car images, organized into subfolders by class (as expected by PyTorch's `ImageFolder`).
- You do **not** need annotation files like `.mat` or `label.labels.txt` - the model is trained directly from the image folder structure.


### 2. Clone the Repository

```bash
git clone https://github.com/Polpoliti/vehicle-classification-cnn.git
cd vehicle-classification-cnn
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

(Consider using a virtual environment.)

---

### 4. Run the Notebook

Launch Jupyter Notebook or your IDE, then open and run:

```
Final - ComputerVision_E2E_CNN_Cars196.ipynb
```

The notebook includes:
- Data loading using PyTorch's ImageFolder structure (no `.mat` or `.txt` files required)
- Custom CNN architecture with residual blocks (no pretrained weights)
- Data augmentation, class balancing, and training logic
- Accuracy tracking and GradCAM visualizations


---

### Tips

- A GPU is highly recommended for training (e.g., Colab Pro or local CUDA setup)
- You can tweak hyperparameters directly in the notebook (batch size, learning rate, MixUp probability, etc.)
- GradCAM visualizations are available at the end of the notebook - enable them to inspect where the model focuses for each prediction
- Training logs and performance metrics (accuracy/loss) are plotted in real time to monitor convergence and overfitting

---

## Experimentation and Challenges

Throughout this project, I experimented with multiple training strategies and faced several real world challenges:

- **Simple CNN architectures** without residuals failed to converge well on 196 classes. Accuracy plateaued early, even with heavy data augmentation.
- Adding **residual connections** significantly improved learning stability and enabled deeper architectures to train effectively.
- I compared **oversampling**, **weighted loss**, and **WeightedRandomSampler** - the sampler approach combined with **label smoothing** produced the most consistent results.
- Without **MixUp**, the model tended to overfit after ~20 epochs. Integrating MixUp with probability 0.5 helped improve generalization.
- Initially, I used a constant learning rate, but switching to **OneCycleLR** with cosine annealing led to smoother convergence and better final performance.
- I observed overfitting around epoch 60-65, which is why I integrated **EarlyStopping** to capture the best checkpoint before performance dropped.

These iterations taught me how small architectural tweaks and the right combination of training techniques can lead to substantial improvements in a deep learning workflow - especially when working without transfer learning.

---

## What I Learned

This project gave me deep, hands-on experience in:

- Building and debugging CNN architectures
- Tackling class imbalance with principled methods
- Designing data augmentation strategies
- Monitoring training behavior and using GradCAM for explainability
- Balancing performance with generalization - not just chasing accuracy

I gained a much stronger intuition around what works (and doesn’t) in deep learning, especially when building everything myself.
