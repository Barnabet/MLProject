# CNN-Based Waste Classification
## Automated Trash Management System

A deep learning project that compares two CNN approaches for automated waste sorting: transfer learning with ResNet-18 vs. training a custom CNN from scratch.

## 🎯 Project Overview

This project implements computer vision-based waste classification to improve recycling efficiency and reduce environmental impact. We evaluate two different CNN architectures on the Kaggle Waste Classification dataset:

1. **Transfer Learning**: Fine-tuned ResNet-18 with ImageNet pretrained weights
2. **From Scratch**: Custom lightweight CNN trained from random initialization

## 📊 Dataset

The project uses a waste classification dataset with two main categories:
- **Organic** (O): Biodegradable waste materials
- **Recyclable** (R): Materials that can be recycled

### Dataset Structure
```
data/
├── train/
│   ├── O/          # Organic waste images
│   └── R/          # Recyclable waste images
└── test/
    ├── O/          # Test organic waste images
    └── R/          # Test recyclable waste images
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- PIL (Pillow)
- numpy

### Installation
```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow numpy
```

### Running the Notebook
1. Ensure your dataset is placed in the `data/` directory following the structure above
2. Open `classification.ipynb` in Jupyter Notebook or VS Code
3. Run all cells sequentially

## 🏗️ Model Architectures

### Model A: ResNet-18 Transfer Learning
- **Base**: ResNet-18 pretrained on ImageNet
- **Approach**: Freeze backbone, fine-tune only the final classifier
- **Parameters**: ~11M (only classifier trainable)
- **Training**: 10 epochs with Adam optimizer
- **Learning Rate**: 1e-3 with StepLR scheduler

### Model B: Custom CNN from Scratch
- **Architecture**: 5-block CNN with progressive channel increase
- **Layers**: Conv2D + BatchNorm + ReLU + MaxPool blocks
- **Parameters**: ~6M total parameters
- **Training**: 20 epochs with AdamW optimizer
- **Learning Rate**: 3e-4 with CosineAnnealingLR scheduler

## 📈 Results

### Performance Metrics
- **Target Accuracy**: ≥90% on test set
- **Evaluation**: Test accuracy, F1-scores per class, confusion matrices
- **Comparison**: Training curves, convergence speed, model efficiency

### Key Findings
- **Transfer Learning** achieves higher accuracy with fewer training epochs
- **Custom CNN** demonstrates strong performance despite training from scratch
- Both models exceed the 90% accuracy target
- Transfer learning shows faster convergence due to ImageNet pretraining

## 📁 Project Structure

```
.
├── classification.ipynb          # Main notebook with complete pipeline
├── README.md                    # This file
├── data/                        # Dataset directory
│   ├── train/                   # Training images
│   └── test/                    # Test images
├── waste_classifier_resnet18.pt # Saved ResNet-18 model weights
└── waste_classifier_scratch.pt  # Saved custom CNN model weights
```

## 🔬 Methodology

1. **Data Preprocessing**:
   - Image resizing to 224×224 pixels
   - Data augmentation (rotation, flipping, random crops)
   - ImageNet normalization

2. **Training Strategy**:
   - 80/20 train/validation split
   - Batch size: 32
   - Early stopping based on validation accuracy
   - Model checkpointing for best weights

3. **Evaluation**:
   - Test set evaluation with held-out data
   - Classification reports with precision, recall, F1-scores
   - Confusion matrices for error analysis
   - Random sample predictions for qualitative assessment

## 🛠️ Key Features

- **Modular Code**: Clean, reusable pipeline structure
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Reproducibility**: Fixed random seeds for consistent results
- **Device Agnostic**: Automatic GPU/MPS/CPU detection
- **Model Persistence**: Automatic saving of best model weights

## 🎯 Objectives & Targets

| Goal | Description | Status |
|------|-------------|--------|
| Model Comparison | ResNet-18 TL vs. scratch CNN | ✅ Completed |
| ≥90% Accuracy | State-of-the-art performance | ✅ Achieved |
| Reusable Pipeline | Modular, documented code | ✅ Implemented |

## 🔮 Future Work

- **Edge Deployment**: Optimize models for mobile/edge devices using TensorRT or CoreML
- **Data Augmentation**: Generate synthetic images for under-represented classes
- **Extended Applications**: Object detection and segmentation for robotic systems
- **Multi-class Extension**: Support for more waste categories (glass, metal, paper, etc.)
- **Real-time Processing**: Implement streaming inference pipeline

## 📊 Visualizations

The notebook includes:
- Training/validation loss and accuracy curves
- Comparative performance bar charts
- Confusion matrices for both models
- Random sample predictions with confidence scores

## 🤝 Contributing

This is an academic project. For improvements or extensions:
1. Fork the repository
2. Create a feature branch
3. Implement changes with proper documentation
4. Submit a pull request

## 📝 License

This project is for educational purposes as part of a Machine Learning course.

## 📞 Contact

For questions about this project, please refer to the course materials or contact the instructor.

---

*Last updated: May 27, 2025*
