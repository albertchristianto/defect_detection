import argparse
import torch
import cv2

from model import get_model
from dataloader import *

def run():
    parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Classification Testing Code by Albert Christianto')
    parser.add_argument('--use_gpu', action='store_true', default = True) 
    parser.add_argument('--input_path', default='E:/Albert_Christianto/Project/defect_detection/ImgClassifier/samples/lvNing_000005.bmp',
                            type=str, metavar='DIR', help='path to train list')
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                                            metavar='N', help='batch size (default: 16)')
    parser.add_argument('--weight_path', default='checkpoint/20221026-0454/img_classifier_best_epoch_160.pth', type=str, metavar='DIR',
                                            help='path to weight of the model')   
    parser.add_argument('--model_type', type=str, default='ResNet50', help='define the model type that will be used: VGG16, ResNet34, ResNet50')
    parser.add_argument('--class_name_path', default='E:/Albert_Christianto/Project/defect_detection/dataset/private_pcb/classes_name.txt', type=str, metavar='DIR',
                                            help='path to weight of the model')
    parser.add_argument('--means_stds_path', default='E:/Albert_Christianto/Project/defect_detection/dataset/private_pcb/mean_stds.txt', type=str, metavar='DIR',
                                            help='path to weight of the model')
    parser.add_argument('--input_size', default=224, type=int, metavar='N',
                                            help='number of epochs to save the model')
    args = parser.parse_args()

    class_name = get_class_name(args.class_name_path)
    means, stds = get_means_stds(args.means_stds_path)
    #BUILDING THE NETWORK
    print('Building {} network'.format(args.model_type))
    cnn_model = get_model(args.model_type, len(class_name), args.input_size, None)
    print('Finish building the network')
    print(cnn_model)

    #load the trained network
    cnn_model.load_state_dict(torch.load(args.weight_path))

    cnn_model.eval()
    img = cv2.imread(args.input_path)
    img = cv2.resize(img, (args.input_size, args.input_size), interpolation=cv2.INTER_CUBIC)
    disp = img.copy()
    img = vgg_preprocess(img, means, stds)
    batch_images= np.expand_dims(img, 0)
    batch_var = torch.from_numpy(batch_images).float()
    #load the model and the criterion in the GPU
    if args.use_gpu:
        cnn_model.cuda()
        batch_var = batch_var.cuda()
    outputs = cnn_model(batch_var)
    _, preds = torch.max(outputs, 1)
    cv2.imshow('the_input', disp)
    print('the image is classified as {}'.format(class_name[int(preds)]))
    cv2.waitKey(0)

if __name__ == '__main__':
    run()
