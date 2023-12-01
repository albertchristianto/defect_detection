import argparse
from model import get_model

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
    parser.add_argument('--model_type', type=str, default='efficientnetv2_m', help='define the model type that will be used')
    args = parser.parse_args()
    #BUILDING THE NETWORK
    print(f'Building {args.model_type} network')
    cnn_model = get_model(args.model_type, 1, 'weights/')
    print('Finish building the network')
    train_param = count_parameters(cnn_model)
    print(f'The training parameters for {args.model_type} is {train_param}')

if __name__ == '__main__':
    main()