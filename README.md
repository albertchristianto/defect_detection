# Defect Detection
This repository contains a project for defect detection. I started this side project to fill my free time. The goal in my mind about this project is how to scale up the project to tackle any field of defect detection using the latest technology in deep learning. I will also provide a simple inference backend system. Finally, I will build this project with Continous Integration and Continous Development in mind.

## Motivation
Defect detection is a broad topic in the industrial world. The semiconductor manufacturing industry needs defect detection systems to support the engineer in delivering various kinds of semiconductor products with good quality. A defect detection system improves the existing quality assurance process and simultaneously cuts costs by reducing the workload of the quality assurance workers. Other industry examples which need this system are the magnetic tile industry, civil-construction industry, electronic device industry, etc. 

## Proposed Methods and Experiments Results
### Public Dataset Used in this project
I use the public defect detection dataset to build and test the performance of the proposed methods. The links for the dataset are:
1. [Magnetic Tile Surface Dataset](https://github.com/abin24/Magnetic-tile-defect-datasets.)
2. Incoming
### Image Classification
The most obvious method for defect detection is image classification. In the deep learning method, there is a lot of neural network architecture, such as VGG, ResNet, EfficientNet, etc. These neural network architectures usually are used for other deep learning methods as their backbone architecture. By using image classification, we can analyze the performance of the backbone architecture and check the deployability of the architecture. You can find the image classifier implementation [here](https://github.com/albertchristianto/defect_detection/tree/main/ImgClassifier).


I will explain about the problem definition here.


Below is the performance comparison table for the [Magnetic Tile Surface Dataset](https://github.com/abin24/Magnetic-tile-defect-datasets.).

| Backbone Name                              | Number of  Training Parameters | Training Epochs | Learning Rate | Accuracy(%) |
| :----------------------------------------- |:------------------------------:|:---------------:|:-------------:| -----------:|
| ResNet34 - Scratch                         |                     21,285,698 |             150 |         0.001 |       97.16 |
| ResNet34 - ImageNet Pre-trained            |                     21,285,698 |             150 |         0.001 |       97.39 |
| ResNet50 - Scratch                         |                     23,512,130 |             150 |         0.001 |       95.14 |
| ResNet50 - ImageNet Pre-trained            |                     23,512,130 |             150 |         0.001 |       98.20 |
| EfficientNet-B0 - Scratch                  |                      4,010,110 |             500 |         0.001 |       95.43 |
| __EfficientNet-B0 - ImageNet Pre-trained__ |                  __4,010,110__ |         __150__ |     __0.001__ |   __98.34__ |

From the performance, we can see that image classification tackles defect detection for the magnetic tile surface. The best backbone architecture for detecting the defect is EfficientNet-B0 achieving 98.34% accuracy, despite having fewer training parameters. Using ImageNet pre-trained weights fastens training times and achieves high accuracy. All the models trained from scratch have lower accuracy. The reason behind this phenomenon is transfer learning. This [article](https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/) explains it clearly.

## Continous Integration and Continous Development
Incoming description from me

## Reference
1. J. Wei, P. Zhu, X. Qian and S. Zhu, "One-stage object detection networks for inspecting the surface defects of magnetic tiles," 2019 IEEE International Conference on Imaging Systems and Techniques (IST), 2019, pp. 1-6, doi: 10.1109/IST48021.2019.9010098.
