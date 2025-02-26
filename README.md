# Medical Image Segmentation Pipeline

This repository provides an end-to-end pipeline for medical image segmentation using deep learning. It is implemented with Python, TensorFlow, OpenCV, and other popular libraries. The project includes two Google Colab notebooks—one for training a segmentation model (`train.ipynb`) and one for making predictions (`Prediction.ipynb`). The pipeline is modular and fully customizable, allowing you to integrate your own segmentation models, adjust file paths, and tweak training parameters as needed.

## Repository Structure

Medical_Image_Segmentation_Pipeline/ ├── dataset/ │ ├── images/ # Original training images │ └── masks/ # Corresponding ground truth segmentation masks ├── segmentation_models/ # Model architecture files (e.g., unet.py, your_custom_model.py) ├── input_images/ # Images for prediction/inference ├── predicted_masks/ # Output directory for predicted segmentation masks ├── save_trained_models/ # Directory where trained models are saved ├── train.ipynb # Notebook for data augmentation, model training, and evaluation ├── Prediction.ipynb # Notebook for loading the model and making predictions ├── README.md # Project documentation (this file) ├── LICENSE # MIT License ├── CONTRIBUTING.md # Guidelines for contributing to this project ├── CHANGELOG.md # List of changes for each version ├── ISSUE_TEMPLATE.md # Template for reporting issues └── PULL_REQUEST_TEMPLATE.md # Template for submitting pull requests


## Requirements

- **Python 3.x**
- **TensorFlow 2.x** – For building, training, and loading segmentation models.
- **OpenCV** – For image loading and preprocessing.
- **NumPy** – For numerical operations.
- **Matplotlib** – For plotting training curves.
- **scikit-learn** – For splitting the dataset into training and validation sets.

> The notebooks are optimized for running on [Google Colab](https://colab.research.google.com/). Google Drive is used for file storage and management.

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/Medical_Image_Segmentation_Pipeline.git
   cd Medical_Image_Segmentation_Pipeline

    Google Drive Setup:
        Place your training dataset in the dataset/ folder:
            Training images in dataset/images/
            Corresponding ground truth masks in dataset/masks/
        For predictions, add your images to the input_images/ folder.
        The notebooks will mount your Google Drive to access these folders, save trained models, and store predicted masks.

    Open the Notebooks in Google Colab:
        Run train.ipynb for training.
        Run Prediction.ipynb for inference.

Usage
1. Training the Model (train.ipynb)

    Configuration & File Paths:
    Set paths and parameters at the start of the notebook. For example:

    MODEL_NAME = "unet"
    MODEL_DIR = "/content/drive/MyDrive/Medical_Image_Segmentation_Pipeline/segmentation_models"
    DATASET_PATH = "/content/drive/MyDrive/Medical_Image_Segmentation_Pipeline/dataset"
    SAVE_MODEL_PATH = "/content/drive/MyDrive/Medical_Image_Segmentation_Pipeline/save_trained_models"
    INPUT_IMAGE_PATH = "/content/drive/MyDrive/Medical_Image_Segmentation_Pipeline/input_images"
    PREDICTED_MASK_PATH = "/content/drive/MyDrive/Medical_Image_Segmentation_Pipeline/predicted_masks"

    Data Loading & Naming Conventions:
    Ensure filenames in dataset/images/ and dataset/masks/ match exactly.

    Data Augmentation & Hyperparameters:
    Functions like random_flip, random_rotate, and adjust_brightness expand your dataset. Hyperparameters such as IMG_HEIGHT, BATCH_SIZE, and EPOCHS are defined here.

    Custom Model Integration:
    Add your custom model file (e.g., my_custom_model.py) in segmentation_models/ and update MODEL_NAME in the notebook.

2. Making Predictions (Prediction.ipynb)

    File Paths & Preprocessing:
    Set paths for the saved model and input images. A preprocessing function resizes and normalizes images.

    Prediction & Saving Masks:
    The notebook loads the trained model, predicts segmentation masks, applies a threshold, resizes to original dimensions, and saves the results.

Data Naming Conventions

    File Name Matching:
    Ensure that images in dataset/images/ and their corresponding masks in dataset/masks/ have exactly the same filename.

Customization

    Model Architecture:
    Switch or add models by placing your custom model file in segmentation_models/ and updating MODEL_NAME.
    Data Augmentation and Hyperparameters:
    Modify augmentation techniques and adjust parameters as needed.
    File Paths and Environment:
    Update paths in the notebooks to match your setup.

Contributing

We welcome contributions! Please see CONTRIBUTING.md for detailed guidelines.
Changelog

Please see CHANGELOG.md for a list of changes and updates.