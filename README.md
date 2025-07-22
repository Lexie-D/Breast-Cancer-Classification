
# Breast Cancer Classification with Ultrasound Images

This project uses a Convolutional Neural Network (CNN) with a pre-trained DenseNet121 backbone to classify breast ultrasound images into **three categories**:

- **Normal**
- **Benign**
- **Malignant**

## Dataset
The dataset used is **[Breast Ultrasound Images Dataset (BUSI)]** from Hugging Face:  
https://huggingface.co/datasets/gymprathap/Breast-Cancer-Ultrasound-Images-Dataset

It consists of:
- 1,578 Images
- 3 Classes: Normal, Benign, Malignant
- Images are grayscale (converted to RGB for CNN)

## Project Workflow

1. **Data Preprocessing**
   - Resize to `(224, 224, 3)`
   - Normalization & Augmentation
   - Train/Validation/Test split (70/15/15)

2. **Model Architecture**
   - Pretrained **DenseNet121**
   - Custom Fully Connected Layers
   - Final Output Layer with `softmax` for 3 categorical classes

3. **Training**
   - Optimizer: Adam with learning_rate = 1e-4
   - Loss: Categorical Crossentropy
   - Metrics: Accuracy, Precision, Recall, F1-Score, AUC

4. **Callbacks**
   - EarlyStopping
   - ReduceLROnPlateau
   - ModelCheckpoint

5. **Evaluation**
   - Accuracy, Loss on Train/Validation/Test
   - Confusion Matrix
   - AUC-ROC Curve
   - Classification Report

## Results

|             |precision  |recall   |f1-score  |support  |
|-------------|-----------|---------|----------|---------|
|     benign  |    0.90   |  0.93   |  0.92    |  134    |
|  malignant  |    0.94   |  0.76   |  0.84    |  63     |
|     normal  |    0.74   |  0.88   |  0.80    |  40     |
|-------------|-----------|---------|----------|---------|
|   accuracy  |           |         |  0.88    |  237    |
|  macro avg  |    0.86   |  0.86   |  0.85    |  237    |
|weighted avg |    0.88   |  0.88   |  0.88    |  237    |


| Set         | Accuracy | Loss   |
|-------------|----------|--------|
| Train       | 0.8882   | 0.2955 |
| Validation  | 0.8701   | 0.3601 |
| Test        | 0.8776   | 0.3595 |

**Macro-Averaged AUC-ROC**: 0.9588

## Files

- `train_val_test_split.py` — Data splitting
- `test.py` — Model evaluation on test set
- `Breast_Cancer_classification.ipynb` — Main notebook
- `.gitignore` — Exclude large/model files

## To Run

1. Upload dataset to Colab
2. Run all cells in the notebook
3. Review saved models and evaluation results

## Author

Lexie-D | [GitHub](https://github.com/Lexie-D)
