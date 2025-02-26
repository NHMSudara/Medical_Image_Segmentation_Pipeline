# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Experimental features for additional data augmentation techniques.
- Integration support for custom segmentation models.
- Enhanced documentation for configuration and file paths.
- Minor improvements to inference preprocessing steps.

### Fixed
- Updated path settings in the Google Colab notebooks.
- Resolved naming convention mismatches between images and masks.

## [1.0.1] - 2025-02-26
### Fixed
- Corrected file path errors in the Prediction.ipynb notebook.
- Minor bug fixes in the data augmentation functions.

## [1.0.0] - 2025-02-20
### Added
- Initial release of the Medical Image Segmentation Pipeline.
- Training notebook (`train.ipynb`) for data augmentation, model training, and evaluation.
- Prediction notebook (`Prediction.ipynb`) for loading trained models and making inferences.
- Repository structure with dedicated folders for datasets, models, documentation, and outputs.
- Basic segmentation model architecture file (`unet.py`) in `segmentation_models/`.
- Setup instructions for using Google Drive with Google Colab.


