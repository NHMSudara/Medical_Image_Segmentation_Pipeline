# Medical Image Segmentation Pipeline

This repository provides an end-to-end pipeline for medical image segmentation using deep learning. Implemented in Python with TensorFlow, OpenCV, and other popular libraries, this project includes two Google Colab notebooks—one for training a segmentation model (`train.ipynb`) and one for making predictions (`Prediction.ipynb`). The pipeline is modular and fully customizable, allowing you to integrate your own segmentation models, adjust file paths, and tweak training parameters as needed.

---

## Repository Structure

```
Medical_Image_Segmentation_Pipeline/
├── .github/
│   ├── ISSUE_TEMPLATE.md
│   └── PULL_REQUEST_TEMPLATE.md
├── dataset/
│   ├── images/                # Original training images
│   └── masks/                 # Corresponding ground truth segmentation masks
├── segmentation_models/       # Model architecture files (e.g., unet.py, my_custom_model.py)
├── input_images/              # Images for prediction/inference
├── predicted_masks/           # Output directory for predicted segmentation masks
├── save_trained_models/       # Directory where trained models are saved
├── docs/                     # Additional documentation (e.g., config_setup.md)
├── train.ipynb                # Notebook for data augmentation, model training, and evaluation
├── Prediction.ipynb           # Notebook for loading the model and making predictions
├── README.md                  # Project documentation (this file)
├── LICENSE                    # MIT License
├── CONTRIBUTING.md            # Guidelines for contributing to this project
└── CHANGELOG.md               # Documentation of changes and release history
```

---

## Requirements

- **Python 3.x**
- **TensorFlow 2.x** – For building, training, and loading segmentation models.
- **OpenCV** – For image loading and preprocessing.
- **NumPy** – For numerical operations.
- **Matplotlib** – For plotting training curves.
- **scikit-learn** – For splitting the dataset into training and validation sets.

> The notebooks are optimized for running on [Google Colab](https://colab.research.google.com/). Google Drive is used for file storage and management.

---

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/NHMSudara/Medical_Image_Segmentation_Pipeline.git
   cd Medical_Image_Segmentation_Pipeline
   ```

2. **Google Drive Setup:**
   - Place your training dataset in the `dataset/` folder:
     - Training images should go into `dataset/images/`
     - Corresponding ground truth masks should be placed in `dataset/masks/`
   - For predictions, add your images to the `input_images/` folder.
   - The notebooks will mount your Google Drive to access these folders, save trained models, and store predicted masks.

3. **Open the Notebooks in Google Colab:**
   - Run `train.ipynb` for training the model.
   - Run `Prediction.ipynb` for inference.

---

## Configuration & File Paths

At the beginning of the notebooks, set paths and parameters according to your directory structure. For example:

```python
MODEL_NAME = "unet"
MODEL_DIR = "/content/drive/MyDrive/Medical_Image_Segmentation_Pipeline/segmentation_models"
DATASET_PATH = "/content/drive/MyDrive/Medical_Image_Segmentation_Pipeline/dataset"
SAVE_MODEL_PATH = "/content/drive/MyDrive/Medical_Image_Segmentation_Pipeline/save_trained_models"
INPUT_IMAGE_PATH = "/content/drive/MyDrive/Medical_Image_Segmentation_Pipeline/input_images"
PREDICTED_MASK_PATH = "/content/drive/MyDrive/Medical_Image_Segmentation_Pipeline/predicted_masks"
```

---

## Data Loading & Naming Conventions

Ensure that filenames in `dataset/images/` and `dataset/masks/` **match exactly** so that each image pairs correctly with its corresponding mask.

**Example Structure:**

```
dataset/
├── images/
│   ├── patient01.png
│   ├── patient02.png
│   └── ...
├── masks/
│   ├── patient01.png
│   ├── patient02.png
│   └── ...
```

**Code Snippet:**

```python
import os

IMAGE_DIR = os.path.join(DATASET_PATH, "images")
MASK_DIR = os.path.join(DATASET_PATH, "masks")

# List and sort image files to ensure proper pairing
img_files = sorted(os.listdir(IMAGE_DIR))

for img_file in img_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, img_file)  # Filenames must match
    # Load image and corresponding mask
```

---

## Data Augmentation & Hyperparameters

To improve model generalization, apply data augmentation techniques such as random flips, rotations, and brightness adjustments.

### Augmentation Functions

```python
import random
import cv2
import numpy as np

def random_flip(image, mask):
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask

def random_rotate(image, mask):
    angle = random.randint(-30, 30)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return image, mask

def adjust_brightness(image):
    factor = 0.7 + random.uniform(0, 0.6)
    return np.clip(image * factor, 0, 255).astype(np.uint8)
```

### Hyperparameter Settings

```python
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 2
EPOCHS = 30
AUGMENTATION_FACTOR = 10  # Number of augmented samples per image
```

---

## Custom Model Integration

To integrate your custom segmentation model:

1. **Create Your Model File:**  
   - Add your custom model file (e.g., `my_custom_model.py`) to the `segmentation_models/` folder.
   - Ensure the file includes a function that builds and compiles the model:

   ```python
   def build_model(input_size=(256, 256, 3)):
       # Define your model architecture
       model = ...  # Custom model code here
       model.compile(optimizer="adam", loss=dice_loss, metrics=[dice_coefficient])
       return model
   ```

2. **Update the Notebook Configuration:**  
   - Change the `MODEL_NAME` variable to your new model name:

   ```python
   MODEL_NAME = "my_custom_model"
   ```

3. **Dynamic Model Import:**

   ```python
   import importlib

   def get_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
       model_module = importlib.import_module(f"segmentation_models.{MODEL_NAME}")
       return model_module.build_model(input_size)
   ```

---

## Usage

### 1. Training the Model (`train.ipynb`)

- **Configuration & File Paths:**  
  Set the paths as shown in the [Configuration & File Paths](#configuration--file-paths) section.

- **Data Augmentation:**  
  The notebook includes functions to perform random flips, rotations, and brightness adjustments to expand your dataset.

- **Training Process:**  
  Run the notebook to train the segmentation model. Training progress (loss, Dice coefficient) is plotted to monitor performance.

- **Saving the Model:**  
  After training, the model is saved to the folder specified by `SAVE_MODEL_PATH`.

### 2. Making Predictions (`Prediction.ipynb`)

- **File Paths & Preprocessing:**  
  Configure paths to the saved model and input images.

- **Prediction Process:**  
  The notebook loads the trained model, preprocesses input images (resizing, normalization), predicts segmentation masks, applies a threshold to generate binary masks, and resizes them back to the original dimensions.

- **Saving Masks:**  
  Predicted masks are saved in the `predicted_masks/` folder.

---

## Data Naming Conventions

- **File Name Matching:**  
  Ensure that images in `dataset/images/` and their corresponding masks in `dataset/masks/` have exactly the same filenames (including extensions) to enable correct pairing.

- **Tips:**
  - Avoid extra spaces or capitalization differences.
  - If mask files have a different extension, adjust the code accordingly.

---

## Customization

This pipeline is designed to be easily adapted:

- **Model Architecture:**  
  Switch or add segmentation models by placing your custom model file in `segmentation_models/` and updating the `MODEL_NAME` variable in both notebooks.

- **Data Augmentation & Hyperparameters:**  
  Modify the augmentation functions or adjust parameters like `IMG_HEIGHT`, `BATCH_SIZE`, and `EPOCHS` as needed.

- **File Paths:**  
  Update the various path variables if you change your file organization or storage method.

---

## Contributing

We welcome contributions to improve this project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

---

## Changelog

For a list of changes, updates, and release history, please refer to [CHANGELOG.md](CHANGELOG.md).

---

## License

This project is licensed under the [MIT License](LICENSE).

