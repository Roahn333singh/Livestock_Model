# Livestock Disease Prediction using MobileNet

## Overview
This project focuses on classifying cow health conditions using a **deep learning model based on MobileNet**. The model predicts three classes of cow conditions with high accuracy:

- 🐄 **Cow Healthy**
- 🐄 **Cow Lumpy** (Lumpy Skin Disease)
- 🐄 **Cow Rinderpest**

Due to limited available datasets, the model currently functions as a **binary classifier for cow images**, but with additional labeled data, it can be extended to multi-class disease detection.

## Dataset
The dataset consists of **859 images** categorized into three different classes:
- 📂 **Training:** 600 images
- 📂 **Validation:** 171 images
- 📂 **Testing:** 88 images

The dataset was split using **splitfolders** with the ratio of **70% training, 20% validation, and 10% testing**.

## Model Architecture
The model is built using **TensorFlow & Keras**, leveraging **MobileNet** as a feature extractor:
- 🧠 **Base Model:** Pre-trained **MobileNet** (with `imagenet` weights, `include_top=False`)
- 🔄 **Global Average Pooling Layer** for feature aggregation
- 🔢 **Fully Connected Dense Layer** (ReLU activation)
- 🎛️ **Dropout Layer** (0.5 dropout rate for regularization)
- 🎯 **Final Output Layer:** Softmax activation for multi-class classification

## Data Augmentation
To enhance model generalization, the training dataset was augmented using:
- 🔍 **Rescaling** (1./255)
- 🔄 **Width & Height Shift** (20%)
- 🔀 **Shear & Zoom Transformations** (20% & 50%)
- 🔄 **Horizontal & Vertical Flipping**
- 🔄 **Rotation (up to 90 degrees)**

## Training Details
- ⚡ **Optimizer:** Adam
- 🎯 **Loss Function:** Categorical Cross-Entropy
- 📦 **Batch Size:** 32
- ⏳ **Epochs:** 10
- 📊 **Evaluation Metrics:** Accuracy

## Model Performance
The model achieved **high accuracy** on the validation and test sets:
- ✅ **Validation Accuracy:** ~95.3%
- ✅ **Test Accuracy:** **96.56%**
- ❌ **Test Loss:** **0.0955**

### Accuracy & Loss Plots
Below are the training and validation accuracy & loss graphs:

![Training and Validation Accuracy](E36A8D12-6EA8-432F-8E95-813A23C00FB1.png)
![Training and Validation Loss](E71D3D40-51BF-4DE0-88A1-6594F150B09C.png)

## How to Use
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install tensorflow numpy matplotlib splitfolders
```

### Training the Model
1. 📥 **Clone the repository:**
```bash
git clone https://github.com/yourusername/livestock-disease-prediction.git
cd livestock-disease-prediction
```

2. 📂 **Prepare the dataset and split into train, val, and test sets:**
```python
import splitfolders
splitfolders.ratio('Data_extracted/SMART_I_H_DATA', output="output", seed=1337, ratio=(.7, 0.2, 0.1))
```

3. 🚀 **Train the model:**
```python
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
```

### Evaluating the Model
📊 **Run the evaluation on the test dataset:**
```python
loss, acc = model.evaluate(test_generator, verbose=1)
print(f"Test accuracy: {acc}")
```

## Future Improvements
- 🔍 **Expand dataset** with real-world images of diseased and healthy cows.
- 🎨 **Implement Grad-CAM** to visualize important regions for predictions.
- 📱 **Convert model to TensorFlow Lite (TFLite)** for mobile deployment.
- 🌍 **Deploy the model using Flask or Streamlit** for easy accessibility.

## Contributors
- 👤 **Your Name Profile ** - (https://github.com/Roahn333singh)

## License
📜 This project is licensed under the MIT License.



Feel free to contribute by improving the dataset or model performance! 🚀




