# 🩺 SkinCancerMultimodal
Multimodal skin cancer classification using dermoscopic images and clinical metadata with a BioGPT-enhanced model. Achieves 94-95% accuracy and >0.85 BCC recall.
# 💡 Overview
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
```
## 🚀 Installation

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```
## 📁 Dataset

This project uses the **PAD-UFES-20** dataset, which includes dermatoscopic images and clinical metadata for skin lesion classification.

- The dataset is publicly available here: [PAD-UFES-20 Dataset](https://www.sciencedirect.com/science/article/pii/S2352340920310209)
- For convenience, the dataset has been included in the `Dataset/` folder of this repository.

> 📌 **Note**: Please ensure that your use of this dataset complies with its [license and usage terms](https://www.sciencedirect.com/science/article/pii/S2352340920310209).

## 🗂️ Dataset Structure

- **Images**: Stored in the following directories:
  - `images/imgs_part_1/`
  - `images/imgs_part_2/`
  - `images/imgs_part_3/`

- **Metadata**: Contained in `metadata.csv`, which includes fields such as:
  - `age`
  - `gender`
  - `lesion_location`
  - `clinical_symptoms` (e.g., itching, bleeding, evolution, etc.)

  ## ⚙️ Configuring File Paths

Before running the project, update the following variables in `Grok_Final.py` to reflect the location of your dataset:

- `DATASET_PATH`: Root directory of the dataset (e.g., `/path/to/your/dataset`)
- `CSV_PATH`: Path to `metadata.csv` (e.g., `/path/to/your/dataset/metadata.csv`)
- `IMAGE_DIRS`: List of image directory paths  
  (e.g., `[/path/to/imgs_part_1, /path/to/imgs_part_2, /path/to/imgs_part_3]`)
- `CHECKPOINT_DIR`: Directory for model checkpoints (e.g., `/path/to/your/Checkpoints`)

### 🧪 Example Modification in `PAD_Final_pr.ipynb`

```python
DATASET_PATH = "/path/to/your/dataset"
CSV_PATH = os.path.join(DATASET_PATH, "metadata.csv")
IMAGE_DIRS = [
    os.path.join(DATASET_PATH, "images/imgs_part_1"),
    os.path.join(DATASET_PATH, "images/imgs_part_2"),
    os.path.join(DATASET_PATH, "images/imgs_part_3")
]
CHECKPOINT_DIR = "/path/to/your/Checkpoints"
```
📌 Tip: Use os.path.join for compatibility across different operating systems.

## 🧠 Models and Checkpoints

Pre-trained models and training checkpoints are available for download via the following Google Drive link:

[Download Models and Checkpoints](https://drive.google.com/drive/folders/1-0pTs4pAeASSRVHxwIscvp31C_LGI2IY?usp=drive_link)

## 💾 Checkpoint Usage

1. Download the checkpoint files and place them in the `Checkpoints/` directory specified by the `CHECKPOINT_DIR` variable.

2. The checkpoint files follow the naming convention:

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/MonkHacker1/SkinCancerMultimodal.git
cd SkinCancerMultimodal
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download the PAD-UFES-20 dataset and place it in your desired directory. Update the DATASET_PATH, CSV_PATH, and IMAGE_DIRS variables in Grok_Final.py accordingly.

4. Download the model checkpoints from the Google Drive link and place them in the Checkpoints directory. Update CHECKPOINT_DIR in Grok_Final.py to match the storage location.
## 🎯 Usage

The single file `Grok_Final.py` contains all the core functionality for:

- Training the model
- Evaluating model performance
- Making single-image predictions

Below are instructions for each of these tasks.
## 🏋️‍♂️ Training

To train the model, run the following command:

```bash
python PAD_final_pr.py
```
This will:
> - Load and preprocess the dataset.

> - Perform 5-fold cross-validation training.

> - Save checkpoints to CHECKPOINT_DIR (e.g., best_model_foldX.pt).
> - ⚠️ **Note:** Training requires a CUDA-enabled GPU (e.g., NVIDIA A100 recommended).  
> If you are using a different GPU, you may need to adjust the following variables in `PAD_final_pr.py`:  
> - `BATCH_SIZE` (default: 16)  
> - `GRAD_ACCUM_STEPS` (default: 2)
## 📊 Evaluation

To evaluate the model on the test set, ensure the `evaluate_test_set` function is called.Yea but it is already called in the code

You can either:

- Modify `PAD_final_pr.py` to run evaluation directly, or  
- Import and call the function from the script.

### Look For this in code:

```python
best_fold = evaluate_test_set(test_df)
```
This evaluation process will:

- Load the best model for each fold from `CHECKPOINT_DIR`.
- Compute test accuracy, weighted F1-score, and class-wise performance metrics.
- Print classification reports to the console.
- Save confusion matrices as plots in the `results/plots/` directory.

---

### ⚙️ How to run evaluation:

- Comment out or skip the training section in `PAD_final_pr.py`.
- Ensure the test dataset is properly prepared and loaded.
- Run the script to execute evaluation.

## 🔍 Single Image Prediction

To predict the class of a single image along with clinical metadata, use the `predict_single_image` function.

### Example usage (from the end of `PAD_final_pr.py`):

```python
image_path = "/path/to/your/image.jpg"
metadata = {
    'age': 0.5,               # Normalized value
    'gender': 'M',
    'fitspatrick': 3,
    'region': 'chest',
    'diameter_1': 0.2,        # Normalized value
    'diameter_2': 0.3,        # Normalized value
    'itch': False,
    'grew': True,
    'hurt': False,
    'changed': True,
    'bleed': False,
    'smoke': False,
    'drink': False,
    'pesticide': False,
    'skin_cancer_history': False
}
best_fold_num = 4            # Best fold (0–4)
test_accuracy = 0.9592       # Accuracy of the best fold

pred_class, probs, report = predict_single_image(image_path, metadata, best_fold_num, test_accuracy)
```
This function returns:

> - pred_class: Predicted class label

> - probs: Prediction probabilities for each class

> - report: Detailed diagnostic report

## 📂 Project Structure
```bash
SkinCancerMultimodal/
├── PAD_final_pr.py           # Main script with all functionality
├── requirements.txt        # Dependencies
├── Checkpoints/            # Directory for model checkpoints
│   ├── best_model_fold0.pt
│   ├── best_model_fold1.pt
│   └── ...
├── data/                   # Directory for dataset (user-defined)
│   ├── metadata.csv
│   ├── images/
│   │   ├── imgs_part_1/
│   │   ├── imgs_part_2/
│   │   └── imgs_part_3/
├── results/                # Directory for output reports and visualizations
│   ├── diagnosis_report_foldX.txt
│   └── plots/
├── README.md               # This file
```

## 📈 Results

The model achieves the following performance on the test set (1200 samples, balanced across classes):

### Best Fold (Fold 5):

- **Test Accuracy:** 95.92%
- **Weighted F1-Score:** 95.90%

### Class-wise Performance (Fold 5):
```bash
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| BCC   | 94%       | 93%    | 93%      |
| SCC   | 98%       | 100%   | 99%      |
| MEL   | 98%       | 100%   | 99%      |
| ACK   | 93%       | 94%    | 94%      |
| NEV   | 95%       | 95%    | 95%      |
| SEK   | 98%       | 93%    | 95%      |
```

Confusion matrices and performance plots are generated during evaluation and saved in the `results/plots/` directory.
## 🤝 Contributing

Contributions are welcome!  

If you find any bugs, want to request features, or improve the documentation, please feel free to:

- Open an issue  
- Submit a pull request  

Thank you for helping improve this project!

## 📄 License

This project is licensed under the MIT License. See the (LICENSE) file for details.

## 📬 Contact

For questions or support, please open an issue on this repository or contact me at [aamirhussainwani1412@gmail.com](mailto:aamirhussainwani1412@gmail.com).










