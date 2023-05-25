import argparse
from model import get_model
from dataloader import getLoader

def count_parameters(model):
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        total_params+=param
    return total_params

def main():
    parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Classification Testing Code by Albert Christianto')
    parser.add_argument('--dataset_root', default='E:\Albert Christianto\Project\defect_detection\dataset\magnetic_tile', type=str, metavar='DIR', help='path to train list')
    parser.add_argument('--model_type', type=str, default='ResNet34', help='define the model type that will be used')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='number of epochs to save the model')
    args = parser.parse_args()
    _, _, class_name = getLoader(args.dataset_root, None, 1)
    #BUILDING THE NETWORK
    print('Building {} network'.format(args.model_type))
    cnn_model = get_model(args.model_type, len(class_name), args.input_size, None)
    print('Finish building the network')
    train_param = count_parameters(cnn_model)
    print(f'The training parameters for {args.model_type} is {train_param}')

if __name__ == '__main__':
    main()