import sys
import os
sys.path.append(os.getcwd())
sys.path.append('model')
sys.path.append('utils')
import argparse
import torch

from model import get_model
from dataloader import getLoader

def validate(args, cnn_model, valLoader, valDatasetSize):
    print('Validating...')
    cnn_model.eval()
    correct = 0
    for i, (img, label) in enumerate(valLoader):
        if args.use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = torch.autograd.Variable(img)
        label = torch.autograd.Variable(label)
        outputs = cnn_model(img)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == label.data)
    val_acc = correct.double() / valDatasetSize
    print('Accuracy: {:.4f}'.format(val_acc))
    cnn_model.train()
    return val_acc

def run():
    parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Classification Testing Code by Albert Christianto')
    parser.add_argument('--use_gpu', action='store_true', default = False) 
    parser.add_argument('--dataset_root', default='D:/Data/DataSet/mask', type=str, metavar='DIR', help='path to train list')
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                                            metavar='N', help='batch size (default: 16)')
    parser.add_argument('--weight_path', default='checkpoint/myVGG16_best.pth', type=str, metavar='DIR',
                                            help='path to weight of the model')   
    parser.add_argument('--model_type', type=str, default='VGG16', help='define the model type that will be used: VGG16,')
    parser.add_argument('--input_size', default=224, type=int, metavar='N',
                                            help='number of epochs to save the model')
    args = parser.parse_args()

    #this is the setting for data augmentation
    transform = {}
    transform['random_horizontal_flips'] = 0.5
    transform['input_width'] = args.input_size
    transform['input_height'] = args.input_size

    #LOADING THE DATASET
    ##validation
    _, valLoader, len_class_name = getLoader(args.dataset_root, transform, args.batch_size)
    valDatasetSize = len(valLoader.dataset)
    print('validation dataset len: {}'.format(valDatasetSize))

    #BUILDING THE NETWORK
    print('Building {} network'.format(args.model_type))
    cnn_model = get_model(args.model_type, len_class_name, args.input_size, None)
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
    validate(args, cnn_model, valLoader, valDatasetSize)

if __name__ == '__main__':
    run()
