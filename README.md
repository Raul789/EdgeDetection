# Edge Detection: Sobel vs Canny vs Deep Learning  
**Data Mining Software Project**  

## Overview  

Edge detection is a fundamental technique in image processing used to identify object boundaries within an image by detecting abrupt changes in pixel intensity. This project compares three edge detection methods:  
- **Sobel Filter**: A traditional gradient-based method.  
- **Canny Edge Detection**: A multi-stage approach known for its precision and robustness.  
- **Holistically-Nested Edge Detection (HED)**: A deep learning-based approach leveraging fully convolutional networks for end-to-end edge prediction.  

The goal is to evaluate the performance of these methods on the **BSDS500 (Berkeley Segmentation Dataset)** using metrics like **precision**, **recall**, and **F1 score**.  

## Dataset Overview  

The **BSDS500 (Berkeley Segmentation Dataset)** is a well-known dataset in computer vision, widely used for evaluating image segmentation and edge detection algorithms.  
The dataset used in this project is available at:
[https://paperswithcode.com/dataset/bsds500](https://paperswithcode.com/dataset/bsds500)

### Dataset Description  
- **Number of Images**: 22,000 natural images of diverse content, ranging from indoor scenes and portraits to outdoor environments and textured surfaces.  
- **Ground Truth Annotations**: Each image has corresponding edge maps that serve as ground truth for evaluating edge detection performance.  

### Dataset Structure  
The dataset is divided into three subsets:  

- **Training Set**: 200 images for training edge detection models.  
- **Validation Set**: Used to fine-tune hyperparameters and avoid overfitting.  
- **Test Set**: Used for final evaluation and performance comparison.  

## Performance Evaluation  

The project evaluates each edge detection method using the following metrics:  
- **Precision**: The proportion of detected edges that are actual edges.  
- **Recall**: The proportion of actual edges detected by the model.  
- **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure of performance.  

## Features  

- **Sobel Edge Detection**: Fast and simple gradient-based technique for detecting vertical and horizontal edges.  
- **Canny Edge Detection**: Multi-stage process involving noise reduction, gradient computation, and edge tracking by hysteresis.  
- **Deep Learning (HED)**: State-of-the-art approach leveraging deep convolutional networks for accurate edge prediction in complex scenes.  
 

## License  

Copyright (c) 2025 Turc Raul & Ioan Oanea  

All rights reserved. This project is for educational purposes only and may not be reused or redistributed without explicit permission.  
