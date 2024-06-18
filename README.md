# Brain Tumor Segmentation using U-Net

## Overview

This project focuses on the segmentation of brain tumors from MRI images using the U-Net neural network. The dataset used consists of MRI images from patients of Chung Buk National University Hospital (CBNUH), specifically the Apparent Diffusion Coefficient (ADC) and Diffusion Weighted Imaging (DWI) modalities. The objective is to accurately segment brain tumors to assist in medical diagnosis and treatment planning.

## Dataset

### Source
The dataset is provided by Chung Buk National University Hospital and includes MRI images from brain tumor patients. The dataset comprises two types of MRI images:
1. **Apparent Diffusion Coefficient (ADC)**
2. **Diffusion Weighted Imaging (DWI)**

### Structure
The dataset is organized as follows:
```
/data
  /train
    /images
      /adc
        adc_image_1.tiff
        adc_image_2.tiff
        ...
      /dwi
        dwi_image_1.tiff
        dwi_image_2.tiff
        ...
    /masks
      mask_1.tiff
      mask_2.tiff
      ...
  /validation
    /images
      /adc
        adc_image_1.tiff
        adc_image_2.tiff
        ...
      /dwi
        dwi_image_1.tiff
        dwi_image_2.tiff
        ...
    /masks
      mask_1.tiff
      mask_2.tiff
      ...
```
- **Images Folder**: Contains the ADC and DWI MRI images.
- **Masks Folder**: Contains the corresponding segmentation masks for the images.

### Preprocessing
Before feeding the images into the U-Net model, the following preprocessing steps are performed:
1. **Normalization**: All MRI images are normalized to ensure uniformity.
2. **Resizing**: Images are resized to a consistent dimension suitable for the U-Net architecture.
3. **Augmentation**: Data augmentation techniques (e.g., rotation, flipping, scaling) are applied to increase the robustness of the model.

## Model

### U-Net Architecture
The U-Net model is a convolutional neural network (CNN) designed for biomedical image segmentation. It consists of two parts:
1. **Encoder (Contracting Path)**: Captures context in the image by downsampling.
2. **Decoder (Expanding Path)**: Enables precise localization by upsampling.

The architecture is illustrated as follows:

```
Input -> [Convolution + ReLU]x2 -> Max Pooling -> [Convolution + ReLU]x2 -> Max Pooling -> ... -> Bottleneck -> ... -> [Convolution + ReLU]x2 -> Up Convolution -> [Convolution + ReLU]x2 -> Up Convolution -> Output
```

### Implementation
The U-Net model is implemented using TensorFlow/Keras. Key layers include:
- Convolutional layers with ReLU activation
- Max pooling layers for downsampling
- Up convolution (transposed convolution) layers for upsampling
- Concatenation layers to merge the encoder and decoder paths

## Training

### Configuration
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy/Dice Coefficient
- **Metrics**: Accuracy, Dice Score
- **Batch Size**: 64
- **Epochs**: 1000

### Training Procedure
1. **Data Splitting**: The dataset is split into training and validation sets.
2. **Model Compilation**: The U-Net model is compiled with the specified optimizer, loss function, and metrics.
3. **Training**: The model is trained on the training set with validation on the validation set.
4. **Checkpointing**: Model checkpoints are saved during training for the best performance based on validation metrics.

## Evaluation

The model's performance is evaluated using the following metrics:
- **Dice Coefficient**: Measures the overlap between the predicted segmentation and the ground truth.
- **IoU (Intersection over Union)**: Measures the intersection over the union of the predicted and ground truth masks.
- **Accuracy**: Measures the overall accuracy of the segmentation.

## Results

The trained U-Net model is expected to provide high-quality segmentations of brain tumors from ADC and DWI MRI images. The evaluation results will include detailed metrics and qualitative visualizations of the segmentation performance.


## Conclusion

This project demonstrates the application of the U-Net neural network for brain tumor segmentation from MRI images. The use of ADC and DWI modalities provides a comprehensive approach to accurately identify and segment brain tumors, aiding in clinical diagnostics and treatment planning.

