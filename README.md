# Defect Detection
This repository contains a project for defect detection. I started this side project to fill my free time. The goal in my mind about this project is how to scale up the project to tackle any field of defect detection using the latest technology in deep learning. 

## Motivation
Defect detection is a broad topic in the industrial world. The semiconductor manufacturing industry needs defect detection systems to support the engineer in delivering various kinds of semiconductor products with good quality. A defect detection system improves the existing quality assurance process and simultaneously cuts costs by reducing the workload of the quality assurance workers. Other industry examples which need this system are the magnetic tile industry, civil-construction industry, electronic device industry, etc. 

## Proposed Methods and Its Results
### Public Dataset Used in this project
I use the public defect detection dataset to build and test the performance of the proposed methods. These are the link for the dataset:
1. [Magnetic Tile Surface Dataset](https://github.com/abin24/Magnetic-tile-defect-datasets.)
2. Incoming
### Image Classification
The most obvious method for defect detection is image classification. In the deep learning method, there is a lot of neural network architecture, such as VGG, ResNet, EfficientNet, etc. These neural network architectures usually are used for other deep learning methods as their backbone architecture. By using image classification, we can analyze the performance of the backbone architecture and check the deployability of the architecture.

Below is the performance comparison table.

| Backbone Name | Learning Rate | Accuracy(%) |
| :------------ |:---------------:| -----:|
| ResNet34 - scratch | 0.00001 | 68.84 |
| ResNet34 - ImageNet pre-trained | 0.001 | 73.88 |
| EfficientNet-B0 - scratch | 0.001 |  |
| EfficientNet-B0 - scratch | 0.0001 |  |
| EfficientNet-B0 - scratch | 0.00001 |  |

Incoming analysis from me

## Continuous Integration - Continuous Development

Incoming description from me

## Reference
1. J. Wei, P. Zhu, X. Qian and S. Zhu, "One-stage object detection networks for inspecting the surface defects of magnetic tiles," 2019 IEEE International Conference on Imaging Systems and Techniques (IST), 2019, pp. 1-6, doi: 10.1109/IST48021.2019.9010098.