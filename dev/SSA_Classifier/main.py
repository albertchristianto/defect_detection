import argparse
from model import get_model
from dataloader import *
from datetime import datetime

import json
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from loguru import logger

LOG_LEVEL = 'TRACE'
SHOW_LOG_EVERY_N_ITERATIONS = 500
USE_GPU = True

def define_load_training_param():
    logger.trace("Define the training-testing parameter")
    parser = argparse.ArgumentParser(description='PyTorch Image Classification Training Code by Albert Christianto')
    parser.add_argument('--dataset_root', required=True, type=str, help='path to the dataset')
    parser.add_argument('--model_type', type=str, default='deeplabv3_mobilenet_v3', help='define the model type that will be used')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='training batch size')
    parser.add_argument('--val_freq', default=5, type=int, help='')
    parser.add_argument('--use_pretrained', action='store_true', default = False)
    parser.add_argument('--checkpoint_dir', default='checkpoint', type=str, help='path to the checkpoint')
    parser.add_argument('--resume', action='store_true', default = False)
    parser.add_argument('--input_size', default=224, type=int, help='number of epochs to save the model')
    parser.add_argument('--horizontal_flip_prob', default=0.5, type=float, help='initial learning rate')
    parser.add_argument('--rotation_value', default=180, type=int, help='number of epochs to save the model')
    parser.add_argument('--mode', type=str, required=True, help='define the model type that will be used')
    parser.add_argument('--weight_path', default=None, type=str, metavar='DIR', help='path to weight of the model')
    args = parser.parse_args()
    logger.trace("Define the augmentation parameter")
    transform = {}
    transform['random_horizontal_flips'] = args.horizontal_flip_prob
    transform['random_rotation'] = [(-1 * args.rotation_value), args.rotation_value]
    transform['input_size'] = args.input_size

    return args, transform

def load_dataset_and_create_dataloader(args, transform):
    logger.trace('Load classes name of the dataset')
    class_name_path = os.path.join(args.dataset_root, 'classes_name.txt')
    transform['class_name'] = get_class_name(class_name_path)
    logger.trace('Load means and standard deviation of the dataset')
    means_stds_path = os.path.join(args.dataset_root, 'mean_stds.txt')
    transform['means'], transform['stds'] = get_means_stds(means_stds_path)
    logger.trace('Get the dataloader')
    trainLoader, valLoader = getLoader(args.dataset_root, transform, args.batch_size)
    logger.info(f'Train dataset len: {len(trainLoader.dataset)}')
    logger.info(f'Validate dataset len: {len(valLoader.dataset)}')
    return trainLoader, valLoader, transform['class_name'], transform['means'], transform['stds']

def create_model(args, len_class_name):
    logger.info(f'Building {args.model_type} network')
    cnn_model = get_model(args.model_type, len_class_name, args.use_pretrained)
    return cnn_model

def resume_or_new_training(args, cnn_model):
    if args.resume:
        train_checkpoints_path = os.path.join(args.checkpoint_dir,'training_checkpoint.pth.tar')
        checkpoint = torch.load(train_checkpoints_path)
        args.lr = checkpoint['last_lr']
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint['n_iter']
        best_epoch = checkpoint['best_epoch']
        best_acc_epoch_val = checkpoint['best_acc_val']
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Resuming training from epoch {start_epoch}')
        return best_epoch, best_acc_epoch_val, start_epoch, start_iter, args, cnn_model
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, current_time)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    return 0, 0.0, 0, 0, args, cnn_model

def create_loss_function_optimizer_lr_scheduler(args, cnn_model):
    criterion = nn.CrossEntropyLoss()#build loss criterion
    optimizer_cnn_model = optim.Adam(cnn_model.parameters(), args.lr)
    patience_param = args.epochs / (args.val_freq * 20)
    if (patience_param <= 1):
        patience_param = 5
        args.val_freq = args.epochs / (patience_param * 20)
        logger.info(f'Using default training strategy validation every {args.val_freq} epoch(s) and with patience param of {patience_param}')
    lr_train_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_cnn_model, mode='max', patience = patience_param)
    return criterion, optimizer_cnn_model, lr_train_scheduler

def compute_accuracy(true_labels, predicted_labels):
    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()

    return torch.mean((true_labels == predicted_labels).float()).item()

def validate(use_gpu, cnn_model, valLoader):
    logger.trace('Validating the performance')
    cnn_model.eval()
    correct = 0
    for i, (img, label) in enumerate(valLoader):
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        outputs = cnn_model(img)['out']
        preds = torch.argmax(outputs, 1)
        this_batch_acc = compute_accuracy(label, preds)
        correct += this_batch_acc
    val_acc = correct / len(valLoader.dataset)
    logger.info(f'The accuracy is {val_acc}')
    cnn_model.train()
    return val_acc

def saving_checkpoint(use_gpu, epoch, n_iter, best_epoch, best_acc_val, args, cnn_model, valLoader, checkpoints_dir, train_checkpoints_path, writer):
    val_acc = validate(use_gpu, cnn_model, valLoader)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    if (val_acc > best_acc_val):
        best_acc_val = val_acc
        best_epoch = epoch
        model_save_filename = os.path.join(checkpoints_dir,'img_segmentation_best_epoch.pth'.format(epoch))
        torch.save(cnn_model.state_dict(), model_save_filename)
    logger.info(f'Saving checkpoint to {train_checkpoints_path}')
    torch.save({'epoch':epoch, 'n_iter':n_iter, 'last_lr': args.lr, 'best_epoch': best_epoch, 'best_acc_val': best_acc_val,
                'model_state_dict':cnn_model.state_dict()}, train_checkpoints_path)
    return best_acc_val, best_epoch, val_acc

def post_training_process(args, best_acc_epoch_val, best_epoch, class_name, means, stds):
    the_text = f'dataset root: {args.dataset_root}\n'
    the_text += f'model_type: {args.model_type}, input_size: {args.input_size}, lr: {args.lr},'
    the_text += f' batch_size: {args.batch_size}, use_pretrained:{args.use_pretrained} \n'
    the_text += f'The best Accuracy is {best_acc_epoch_val} at epoch {best_epoch}'
    the_text_path = os.path.join(args.checkpoint_dir,'train_results.txt')
    the_file = open(the_text_path, 'w')
    the_file.write(the_text)
    the_file.close()
    logger.info(the_text)

    the_text_path = os.path.join(args.checkpoint_dir, f'{args.model_type}_Segmentation.json')
    dictionary = {}
    dictionary['input_size'] = str(args.input_size)
    dictionary['means'] = []
    for each in means:
        dictionary['means'].append(str(each))
    dictionary['stds'] = []
    for each in stds:
        dictionary['stds'].append(str(each))
    dictionary['class_name'] = class_name
    json_object = json.dumps(dictionary, indent=4)
    with open(the_text_path, "w") as outfile:
        outfile.write(json_object)

def train(args, class_name, means, stds, cnn_model, trainLoader, valLoader):
    best_epoch, best_acc_epoch_val, start_epoch, n_iter, args, cnn_model = resume_or_new_training(args, cnn_model)
    write_class_name(os.path.join(args.checkpoint_dir,'classes_name.txt'), class_name)
    write_means_stds(os.path.join(args.checkpoint_dir,'means_stds.txt'), means, stds)
    train_checkpoints_path = os.path.join(args.checkpoint_dir,'training_checkpoint.pth.tar')
    writer = SummaryWriter(log_dir=args.checkpoint_dir)
    criterion, optimizer_cnn_model, lr_train_scheduler = create_loss_function_optimizer_lr_scheduler(args, cnn_model)
    use_gpu = USE_GPU and torch.cuda.is_available()
    if use_gpu:
        cnn_model.cuda()
    cnn_model.train()
    for epoch in range(start_epoch, args.epochs):
        running_corrects = 0
        running_corrects_class = 0
        for i, (img, label, seglabel) in enumerate(trainLoader):
            #load all the data in GPU
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
                seglabel = seglabel.cuda()
            class_pred, seg_pred = cnn_model(img)#inference the input
            _, class_pred = torch.max(class_pred, 1)#get the training prediction
            loss = criterion(seg_pred[1], seglabel) + criterion(class_pred, label)#compute the loss#compute the loss
            optimizer_cnn_model.zero_grad() #set gradient to zero
            loss.backward()#compute the gradient
            optimizer_cnn_model.step() #update the model
            seg_pred = torch.argmax(seg_pred, 1)
            running_corrects += compute_accuracy(seglabel, seg_pred)
            running_corrects_class += torch.sum(class_pred == label.data)

            writer.add_scalar('Loss', loss.item(), n_iter)
            if ((n_iter % SHOW_LOG_EVERY_N_ITERATIONS) == 0):
                logger.info('[Epoch {}/{}] Iteration: {}. Loss: {}'.format(epoch, args.epochs, n_iter, loss.item()))
            n_iter += 1
        epoch_class_acc = running_corrects_class.double() / len(trainLoader.dataset)
        epoch_acc = running_corrects / len(trainLoader.dataset)
        writer.add_scalar('Accuracy/train_class', epoch_class_acc, epoch)
        writer.add_scalar('Accuracy/train_seg', epoch_acc, epoch)
        if ((epoch + 1) % args.val_freq) != 0:
            continue
        args.lr = optimizer_cnn_model.param_groups[0]['lr']
        best_acc_epoch_val, best_epoch, val_acc = saving_checkpoint(use_gpu, epoch, n_iter, best_epoch, best_acc_epoch_val, args, cnn_model, 
            valLoader, args.checkpoint_dir, train_checkpoints_path, writer)
        lr_train_scheduler.step(val_acc)
    post_training_process(args, best_acc_epoch_val, best_epoch, class_name, means, stds)

def test(args, cnn_model, valLoader):
    cnn_model.load_state_dict(torch.load(args.weight_path))
    use_gpu = USE_GPU and torch.cuda.is_available()
    if use_gpu:
        cnn_model.cuda()
    validate(use_gpu, cnn_model, valLoader)

if __name__ == '__main__':
    args, transform = define_load_training_param()
    trainLoader, valLoader, class_name, means, stds = load_dataset_and_create_dataloader(args, transform)
    cnn_model = create_model(args, len(class_name))
    if args.mode == 'train':
        train(args, class_name, means, stds, cnn_model, trainLoader, valLoader)
    elif args.mode == 'test':
        test(args, cnn_model, valLoader)
    else:
        logger.warning('Check your mode!')

