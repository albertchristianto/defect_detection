# Defect Detection
This repository contains a project for defect detection. I started this side project to fill my free time. The goal in my mind about this project is how to scale up the project to tackle any field of defect detection using the latest technology in deep learning. Finally, I build this project with Continous Integration and Continous Delivery in mind.

## Motivation
Defect detection is a broad topic in the industrial world. The semiconductor manufacturing industry needs defect detection systems to support the engineer in delivering various kinds of semiconductor products with good quality. A defect detection system improves the existing quality assurance process and simultaneously cuts costs by reducing the workload of the quality assurance workers. Other industry examples which need this system are the magnetic tile industry, civil-construction industry, electronic device industry, etc. 

## Proposed Methods and Experiments Results
### Public Dataset Used in This Project
I use the public defect detection dataset to build and test the performance of the proposed methods. The links for the dataset are:
1. [Magnetic Tile Surface Dataset](https://github.com/abin24/Magnetic-tile-defect-datasets.)
2. [NHA12D](https://github.com/ZheningHuang/NHA12D-Crack-Detection-Dataset-and-Comparison-Study)
3. Incoming
### Image Classification
The most obvious method for defect detection is image classification. In the deep learning method, there is a lot of neural network architecture, such as VGG, ResNet, EfficientNet, etc. These neural network architectures usually are used for other deep learning methods as their backbone architecture. By using image classification, we can analyze the performance of the backbone architecture and check the deployability of the architecture. You can find the image classifier implementation and the experiment reports in [here](https://github.com/albertchristianto/defect_detection/tree/main/ImgClassifier). 

## Continous Integration and Continous Delivery
The development and deployment of the deep learning research are very hard and at a high cost however we can make it more easier and at lower cost By preparing the pipeline for doing research and deploying the research product. [PyTorch](https://pytorch.org/)-[Onnxruntime](https://onnxruntime.ai/)-[TensorRT](https://developer.nvidia.com/tensorrt) is the best strategy to build the CI/CD pipeline as far as I have worked on deep learning-computer vision field. The research part for this repository will use [PyTorch](https://pytorch.org/) as the deep learning framework. Then, I convert the weights of the deep learning network into [Onnxruntime](https://onnxruntime.ai/)'s weights format. Using [TensorRT](https://developer.nvidia.com/tensorrt)'s onnxparser convert the [Onnxruntime](https://onnxruntime.ai/)'s weights format into [TensorRT](https://developer.nvidia.com/tensorrt)'s weights format. With [TensorRT](https://developer.nvidia.com/tensorrt) optimization, the inference performance will be greatly increased.

## Reference
1. J. Wei, P. Zhu, X. Qian and S. Zhu, "One-stage object detection networks for inspecting the surface defects of magnetic tiles," 2019 IEEE International Conference on Imaging Systems and Techniques (IST), 2019, pp. 1-6, doi: 10.1109/IST48021.2019.9010098.
2. Huang, Z., Chen, W., & Brilakis, I. (2022). NHA12D: A New Pavement Crack Dataset and a Comparison Study Of Crack Detection Algorithms. arXiv. https://doi.org/10.48550/arXiv.2205.01198
