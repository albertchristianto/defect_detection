# ImageClassifier
This repository is a re-implementation of the image classification problems using deep learning. You can train an image classifier model using your custom dataset with it. Here are some steps for you:
1. Prepare and organize your dataset similar to this structure below.
```
__YOUR_DATASET__
├── CLASS_A
│   ├── SUBFOLDER_A_CLASS_A
│   │   ├── img_subfolder_a_class_a_001.jpg
│   │   ├── ...
│   │   └── img_subfolder_a_class_a_100.jpg
│   └── SUBFOLDER_B_CLASS_A
│       ├── img_subfolder_b_class_a_001.jpg
│       ├── ...
│       └── img_subfolder_b_class_a_100.jpg
├── CLASS_B
│   ├── SUBFOLDER_A_CLASS_B
│   │   ├── img_subfolder_a_class_b_001.jpg
│   │   ├── ...
│   │   └── img_subfolder_a_class_b_100.jpg
│   └── SUBFOLDER_B_CLASS_B
│       ├── img_subfolder_b_class_b_001.jpg
│       ├── ...
│       └── img_subfolder_b_class_b_100.jpg
└── tools
    ├── compute_mean_stds.py
    └── CreateSplitTxt.py
```
2. For the tools folder, you can copy it from [here](https://github.com/albertchristianto/defect_detection/tree/main/dataset/road_crack/tools).
3. Execute these two python files with this command
```
cd __YOUR_DATASET__/tools
python CreateSplitTxt.py
python compute_mean_stds.py
```
4. Train your model 
```
python main.py --mode train --model_type "CHOOSE_MODEL_TYPE" --dataset_root "YOUR_DATASET_PATH" 
```
5. Test your model 
```
python main.py --mode test --model_type "CHOOSE_MODEL_TYPE" --dataset_root "YOUR_DATASET_PATH" --weight_path "YOUR_WEIGHTS_PATH"
```
6. Use your model on a single image
```
python demo.py --input_path "INPUT_IMAGE_PATH" --model_type "CHOOSE_MODEL_TYPE" --weight_path "YOUR_WEIGHTS_PATH" --class_name_path "CLASSES_NAME_PATH" --means_stds_path "MEANS_STDS_PATH"
```
7. Convert a PyTorch weights format to an Onnx weights format
```
python convert2onnx.py --model_type "CHOOSE_MODEL_TYPE" --weight_path "YOUR_WEIGHTS_PATH" --class_name_path "CLASSES_NAME_PATH" --output_weight_path "OUTPUT_WEIGHTS_PATH"
```
8. Check the numbers of the model's training parameters
```
python check_num_train_param.py --model_type "CHOOSE_MODEL_TYPE" --dataset_root "YOUR_DATASET_PATH" 
```

## Experiment Report
### Experiment for Magnetic Tile Surface Dataset
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

### Experiment for NHA12D Dataset
Road/pavement crack detection uses the same defect detection scheme, patch defect detector. However, the size of the patch is a fixed value(224x224). 

Below is the performance comparison table for the [NHA12D Dataset](https://github.com/ZheningHuang/NHA12D-Crack-Detection-Dataset-and-Comparison-Study).

| Backbone Name                              | Number of  Training Parameters | Accuracy(%) |
| :----------------------------------------- |:------------------------------:| -----------:|
|   ResNet34 - ImageNet Pre-trained          |                         21 M   |     90.62   |
| ResNet50 - ImageNet Pre-trained            |                           23 M |       91.37 |
| __EfficientNet-B0 - ImageNet Pre-trained__ |                        __4 M__ |   __92.01__ |

EfficientNet-B0 achieves the best performance with 92.01% accuracy. However, while monitoring the training process, I notice all the models haven't converged yet. This phenomenon is an implementation problem from the image classification repository in the learning rate part. I will check with the training code of the image classification and update the results.

## Reference
1. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ArXiv. https://doi.org/10.48550/arXiv.1905.11946