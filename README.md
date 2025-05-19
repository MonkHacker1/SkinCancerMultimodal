# SkinCancerMultimodal
Multimodal skin cancer classification using dermoscopic images and clinical metadata with a BioGPT-enhanced model. Achieves 94-95% accuracy and >0.85 BCC recall.
# Overview
This repository contains a Multimodal Deep Learning System implemented in a single Python file (PAD_Final_pr.ipynb) for skin cancer classification. The system integrates dermatoscopic images and clinical metadata (text) to classify skin lesions into six categories: Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), Melanoma (MEL), Actinic Keratosis (ACK), Nevus (NEV), and Seborrheic Keratosis (SEK). The model combines EfficientNet-B3 for image processing, BioGPT for text encoding, and a co-attention mechanism for multimodal fusion, achieving high accuracy on the PAD-UFES-20 dataset.
This project was developed as part of a Master's thesis in Computer Science, focusing on advanced deep learning techniques for medical imaging and clinical data analysis.
## 🔍 Key Features

- **Multimodal Architecture**: Fuses image and text data using EfficientNet-B3, BioGPT, and co-attention.
- **Data Augmentation**: Employs CutMix, MixUp, and Albumentations for robust training.
- **Class Imbalance Handling**: Uses oversampling and Focal Loss to address dataset imbalances.
- **Evaluation Metrics**: Provides accuracy, weighted F1-score, and class-wise performance with confusion matrices.
- **Prediction Pipeline**: Supports single-image predictions with clinical metadata, generating detailed diagnostic reports.

## 📦 Requirements

To run the project, ensure you have the following dependencies installed:

```bash
python>=3.8  
torch>=1.10.0  
transformers>=4.20.0  
timm>=0.5.0  
albumentations>=1.0.0  
pandas>=1.3.0  
numpy>=1.20.0  
matplotlib>=3.4.0  
seaborn>=0.11.0  
scikit-learn>=0.24.0  
Pillow>=8.3.0  
tqdm>=4.60.0

## 🚀 Installation

Install the dependencies using the following command:

```bash
pip install -r requirements.txt

## 📁 Dataset

This project uses the **PAD-UFES-20** dataset, which includes dermatoscopic images and clinical metadata for skin lesion classification.

- The dataset is publicly available here: [PAD-UFES-20 Dataset](https://www.sciencedirect.com/science/article/pii/S2352340920310209)
- For convenience, the dataset has been included in the `Dataset/` folder of this repository.

> 📌 **Note**: Please ensure that your use of this dataset complies with its [license and usage terms](https://www.sciencedirect.com/science/article/pii/S2352340920310209).





