import sys
import os
sys.path.append(os.getcwd())
sys.path.append('model')
sys.path.append('utils')
import argparse
import torch

from model import get_model
from dataloader import getLoader

def validate(use_gpu, cnn_model, valLoader):
    valDatasetSize = len(valLoader.dataset)
    print('Validating...')
    cnn_model.eval()
    correct = 0
    for i, (img, label) in enumerate(valLoader):
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        outputs = cnn_model(img)
        _, preds = torch.max(outputs, 1)
        correct_array = preds == label.data
        correct += torch.sum(correct_array)
    val_acc = correct.double() / valDatasetSize
    print('Accuracy: {:.4f}'.format(val_acc))
    cnn_model.train()
    return val_acc

def run():
    parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Classification Testing Code by Albert Christianto')
    parser.add_argument('--use_gpu', action='store_true', default = False) 
    parser.add_argument('--dataset_root', default='E:\Albert Christianto\Project\defect_detection\dataset\magnetic_tile', type=str, metavar='DIR', help='path to train list')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='batch size (default: 16)')
    parser.add_argument('--weight_path', default='checkpoint/20221007-0159/img_classifier_best_epoch_49.pth', type=str, metavar='DIR', help='path to weight of the model')   
    parser.add_argument('--model_type', type=str, default='EfficientNetB0', help='define the model type that will be used')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='number of epochs to save the model')
    parser.add_argument('--means_stds', type=str, default='mt_means_stds', help='define means and stds that will be used')
    args = parser.parse_args()

    #this is the setting for data augmentation
    transform = {}
    transform['random_rotation'] = [-20, 20]
    transform['random_horizontal_flips'] = 0.5
    transform['input_width'] = args.input_size
    transform['input_height'] = args.input_size
    transform['means_stds'] = args.means_stds

    #LOADING THE DATASET
    ##validation
    _, valLoader, class_name = getLoader(args.dataset_root, transform, args.batch_size)
    valDatasetSize = len(valLoader.dataset)
    print('validation dataset len: {}'.format(valDatasetSize))

    #BUILDING THE NETWORK
    print('Building {} network'.format(args.model_type))
    cnn_model = get_model(args.model_type, len(class_name), args.input_size, None)
    print('Finish building the network')
    print(cnn_model)

    #load the trained network
    cnn_model.load_state_dict(torch.load(args.weight_path))

    #load the model and the criterion in the GPU
    if args.use_gpu:
        cnn_model.cuda()

    ##Last validation------------------------------------------------------------------------- 
    #validate
    #set cnn_model on the val mode
    validate(args.use_gpu, cnn_model, valLoader)

if __name__ == '__main__':
    run()
