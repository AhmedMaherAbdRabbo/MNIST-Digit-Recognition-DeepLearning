# MNIST Handwritten Digit Recognition using Deep Learning 🔢

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.20%25-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

## 📊 Project Overview

This project implements a **state-of-the-art handwritten digit recognition system** using Convolutional Neural Networks (CNN). By leveraging deep learning techniques and the famous MNIST dataset, we develop a high-performance model that accurately classifies handwritten digits with **99.20% accuracy**, demonstrating advanced computer vision capabilities for real-world applications.

### 🎯 Key Objectives
✔ Implement robust CNN architecture with BatchNormalization and Dropout layers

✔ Achieve high classification accuracy (99.20%+) on MNIST digit recognition

✔ Develop comprehensive data preprocessing pipeline with normalization

✔ Create interactive visualization tools for model analysis and predictions

✔ Process large-scale image dataset (70,000 handwritten digit samples)

✔ Implement advanced training techniques with callbacks and learning rate scheduling

✔ Build modular, reusable code architecture for production deployment

✔ Provide extensive model evaluation with confusion matrices and classification reports

## 📁 Dataset Information

### Data Source

- **Training Dataset**: 42,000 labeled digit images (28x28 pixels)
- **Test Dataset**: 28,000 unlabeled images for competition submission
- **Format**: CSV files with flattened pixel values (784 features + 1 label)


## 🗂️ Project Structure

```
MNIST-Digit-Recognition-DeepLearning/
│
├── 📁 data/
│   └── MNIST-Handwritten Digit.zip    # Raw MNIST dataset (Download required)
│
├── 📁 models/                         # Trained Models (Generated after training)
│   └── mnist_digit_recognizer_acc_99.20.h5  # Best performing CNN model
│
├── 📁 notebooks/
│   ├── MNIST_Handwritten_Digit_Recognition_DeepLearning.ipynb  # Main analysis notebook
│   ├── mnist_handwritten_digit_recognition_deeplearning.py     # Python script version
│   └── mnist_digit_recognizer_acc_99.20.h5                    # Model backup
│
│
├── README.md
├── requirements.txt
└── LICENSE
```

## 🔍 Key Features & Functionality

### Data Processing Pipeline
1. **Image Reshaping**: Convert flattened pixels to 28×28×1 format
2. **Pixel Normalization**: Scale values from [0-255] to [0-1] range
3. **Label Encoding**: Convert digits to one-hot categorical format
4. **Data Splitting**: Strategic 80-20 train-validation split with stratification

### Model Training Features
- **Advanced Callbacks**: EarlyStopping + ReduceLROnPlateau
- **Batch Processing**: Efficient 128-sample batch training
- **Progress Monitoring**: Real-time accuracy and loss tracking
- **Model Checkpointing**: Automatic best model preservation

### Evaluation & Testing Features
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visual Analysis**: Confusion matrices with heatmap visualization
- **Prediction Showcase**: Side-by-side true vs predicted comparisons
- **Error Analysis**: Detailed examination of misclassified samples
- **Interactive Testing**: Functions for single and batch predictions


## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📞 Contact

**Ahmed Maher Abd Rabbo**
- 💼 [LinkedIn](https://www.linkedin.com/in/ahmed-maherr/)
- 📊 [Kaggle](https://kaggle.com/ahmedmaherabdrabbo)
- 📧 Email: ahmedbnmaher1@gmail.com
- 💻 [GitHub](https://github.com/AhmedMaherAbdRabbo)


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.