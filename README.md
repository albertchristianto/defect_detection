# Defect Detection
This repository contains a project for defect detection. I started this side project to fill my free time. The goal in my mind about this project is how to scale up the project to tackle any field of defect detection using the latest technology in deep learning. I will also provide a simple inference backend system. Finally, I will build this project with Continous Integration and Continous Delivery in mind.

## Motivation
Defect detection is a broad topic in the industrial world. The semiconductor manufacturing industry needs defect detection systems to support the engineer in delivering various kinds of semiconductor products with good quality. A defect detection system improves the existing quality assurance process and simultaneously cuts costs by reducing the workload of the quality assurance workers. Other industry examples which need this system are the magnetic tile industry, civil-construction industry, electronic device industry, etc. 

## Proposed Methods and Experiments Results
### Public Dataset Used in This Project
I use the public defect detection dataset to build and test the performance of the proposed methods. The links for the dataset are:
1. [Magnetic Tile Surface Dataset](https://github.com/abin24/Magnetic-tile-defect-datasets.)
2. [NHA12D](https://github.com/ZheningHuang/NHA12D-Crack-Detection-Dataset-and-Comparison-Study)
3. Incoming
### Image Classification
The most obvious method for defect detection is image classification. In the deep learning method, there is a lot of neural network architecture, such as VGG, ResNet, EfficientNet, etc. These neural network architectures usually are used for other deep learning methods as their backbone architecture. By using image classification, we can analyze the performance of the backbone architecture and check the deployability of the architecture. You can find the image classifier implementation [here](https://github.com/albertchristianto/defect_detection/tree/main/ImgClassifier).

#### Experiment for Magnetic Tile Surface Dataset
Due to the lack of images, the problem definition for magnetic tile surface defect detection is the defect detector will classify a magnetic tile surface by sliding across the image with the size of the defect detector's input. This scheme also helps increase the detection accuracy with a smaller model. This defect detection scheme is a patch defect detector. The patch defect detector's input size depends on the minimum side of the input image. Check the demonstration of this scheme [here](https://docs.google.com/presentation/d/1pR1xuDoaAntRRu5F9N6TAmnNQoNNMkn3hGWH6LbT5PU/edit#slide=id.g16623a9b199_0_183).


Below is the performance comparison table for the [Magnetic Tile Surface Dataset](https://github.com/abin24/Magnetic-tile-defect-datasets.).

| Backbone Name                              | Number of  Training Parameters | Accuracy(%) |
| :----------------------------------------- |:------------------------------:| -----------:|
| ResNet34 - Scratch                         |                           21 M |       97.16 |
| ResNet34 - ImageNet Pre-trained            |                           21 M |       97.39 |
| ResNet50 - Scratch                         |                           23 M |       95.14 |
| ResNet50 - ImageNet Pre-trained            |                           23 M |       98.20 |
| EfficientNet-B0 - Scratch                  |                            4 M |       95.43 |
| __EfficientNet-B0 - ImageNet Pre-trained__ |                        __4 M__ |   __98.34__ |

The performance table above shows that image classification tackles defect detection for the magnetic tile surface. The best backbone architecture for detecting the defect is EfficientNet-B0 achieving 98.34% accuracy, despite having fewer training parameters. Using ImageNet pre-trained weights fastens training times and achieves high accuracy. All the models trained from scratch have lower accuracy. The reason behind this phenomenon is transfer learning. This [article](https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/) explains it clearly.

#### Experiment for NHA12D Dataset
Road/pavement crack detection uses the same defect detection scheme, patch defect detector. However, the size of the patch is a fixed value(224x224). 

Below is the performance comparison table for the [NHA12D Dataset](https://github.com/ZheningHuang/NHA12D-Crack-Detection-Dataset-and-Comparison-Study).

| Backbone Name                              | Number of  Training Parameters | Accuracy(%) |
| :----------------------------------------- |:------------------------------:| -----------:|
|   ResNet34 - ImageNet Pre-trained          |                         21 M   |     90.62   |
| ResNet50 - ImageNet Pre-trained            |                           23 M |       91.37 |
| __EfficientNet-B0 - ImageNet Pre-trained__ |                        __4 M__ |   __92.01__ |

EfficientNet-B0 achieves the best performance with 92.01% accuracy. However, while monitoring the training process, I notice all the models haven't converged yet. This phenomenon is an implementation problem from the image classification repository in the learning rate part. I will check with the training code of the image classification and update the results.

## Continous Integration and Continous Delivery
Incoming description from me. I am still building a C++ library to help me build an inference backend system. I am using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) as the reference for this C++ library.

## Reference
1. J. Wei, P. Zhu, X. Qian and S. Zhu, "One-stage object detection networks for inspecting the surface defects of magnetic tiles," 2019 IEEE International Conference on Imaging Systems and Techniques (IST), 2019, pp. 1-6, doi: 10.1109/IST48021.2019.9010098.
2. Huang, Z., Chen, W., & Brilakis, I. (2022). NHA12D: A New Pavement Crack Dataset and a Comparison Study Of Crack Detection Algorithms. arXiv. https://doi.org/10.48550/arXiv.2205.01198
