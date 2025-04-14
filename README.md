# AAI-501-01
Introduction to Artificial Intelligence (AAI-501-01) - Assignment7.1 - Final Project 

This project explores the effectiveness of three convolutional neural network (CNN) architectures for fashion image classification using the FashionMNIST dataset (Zalando Research, 2017) and a set of real-world images collected from Google. The first model, a baseline CNN, was trained on raw grayscale images. The second model used the same architecture but incorporated real-time data augmentation to improve robustness. The third model introduced a novel edge-focused approach by training on Canny edge-detected images (Canny, 1986) to prioritize structural features over pixel-level detail.


# Data Source: https://www.kaggle.com/datasets/zalando-research/fashionmnist

# Data Preprocessing
Images were loaded from the FashionMNIST training CSV file and reshaped from flat 784-dimensional vectors into 28×28 grayscale matrices. Pixel values were normalized to the [0, 1] range to improve gradient flow and training convergence. Labels were one-hot encoded for use with categorical cross entropy loss during model compilation.

To test this model on real-world inputs, 20 clothing images were manually sourced from Google. These images were resized to 28×28 pixels using bilinear interpolation and converted to grayscale to match the dimensions and style of the original training data. Like the FashionMNIST images, the real-world images were normalized to the [0, 1] range and reshaped to the required (28, 28, 1) input format.


# Model 1: Baseline CNN 
The first method involved implementing a standard convolutional neural network (CNN) trained on the raw, grayscale FashionMNIST images without any augmentation or feature transformation

The CNN architecture consisted of two convolutional layers followed by max pooling and dropout. The first convolutional layer used 32 filters with a 3×3 kernel and ReLU activation. This was followed by 2×2 max pooling to reduce spatial dimensions and a dropout layer with a rate of 0.3 to prevent overfitting. The second convolutional layer used 64 filters, followed by another max pooling and dropout block.

# Model 2: Augmented CNN
The second method built upon the baseline CNN by introducing real-time data augmentation to increase model robustness and reduce overfitting. The core architecture remained the same, but the input data was dynamically augmented with small affine transformations during training.

As with the baseline model, the FashionMNIST images were normalized to the [0, 1] range and reshaped to (28, 28, 1). Since Keras’ `ImageDataGenerator` expects 3-channel input for augmentation, each image was duplicated across the channel dimension, converting it into a 3-channel grayscale format (28, 28, 3).

The CNN architecture was identical to the baseline, the key difference was in the training input. Using `ImageDataGenerator` (Chollet, 2015), each training batch was randomly transformed using:
- Rotation (±10 degrees)
- Width and height shift (±10%)
- Zoom (±10%)
Padding was applied using `fill_mode='constant'` and `cval=0.0` to preserve the black background in augmented images. These transformations simulated real-world variability in camera angle, position, and scale, encouraging the model to learn more generalizable features. (LeCun, Bengio, & Hinton, 2015)

# Model 3: Edge-Based CNN

The third method introduced a novel approach by transforming all input images into binary edge maps using Canny edge detection before training (Canny, 1986). The intention was to teach the CNN to classify clothing items based solely on structural features (such as contours and outlines), eliminating dependencies on pixel intensities and textures. This method explored the role of abstraction in improving real-world generalization.

FashionMNIST images were first normalized and reshaped as in previous models. They were then converted to 8-bit grayscale and passed through OpenCV’s `Canny()` function with thresholds set to 100 and 200. The resulting binary edge maps retained only the key edges and silhouette of each item. These images were then normalized to the [0, 1] range and reshaped into (28, 28, 1) tensors.



