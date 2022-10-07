import os
import argparse
from datetime import datetime

import torch
torch.manual_seed(17)
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from model import get_model
from dataloader import getLoader
from utils import *
from test import validate

SHOW_LOG_EVERY_N_ITERATIONS = 10000

def Saving_Checkpoint(epoch, n_iter, best_epoch, best_acc_val, args, cnn_model, valLoader, valDatasetSize, checkpoints_dir, train_checkpoints_path, writer):
    val_acc = validate(args, cnn_model, valLoader, valDatasetSize)
    writer.add_scalar('Accuracy/val_set', val_acc, epoch)
    if (val_acc > best_acc_val):
        best_acc_val = val_acc
        best_epoch = epoch
        model_save_filename = os.path.join(checkpoints_dir,'img_classifier_best_epoch_{}.pth'.format(epoch))
        torch.save(cnn_model.state_dict(), model_save_filename)
    print('Saving checkpoint...')
    torch.save({'epoch':epoch,
                'n_iter':n_iter,
                'best_epoch': best_epoch,
                'best_acc_val': best_acc_val,
                'model_state_dict':cnn_model.state_dict()}, train_checkpoints_path)
    return best_acc_val, best_epoch

def run():
    parser = argparse.ArgumentParser(description='PyTorch Image Classification Training Code by Albert Christianto')
    parser.add_argument('--use_gpu', action='store_true', default = True) 
    parser.add_argument('--dataset_root', default='E:\Albert Christianto\Project\defect_detection\dataset\magnetic_tile', type=str, metavar='DIR', help='path to train list')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                                            help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                                            metavar='LR', help='initial learning rate')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='batch size (default: 32)')
    parser.add_argument('--checkpoint_dir', default='checkpoint', type=str, metavar='DIR',
                                            help='path to save tensorboard log and weight of the model')   
    parser.add_argument('--resume', action='store_true', default = False)
    parser.add_argument('--model_type', type=str, default='ResNet34', help='define the model type that will be used')
    parser.add_argument('--means_stds', type=str, default='mt_means_stds', help='define means and stds that will be used')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='number of epochs to save the model')
    parser.add_argument('--save_freq', default=5, type=int, metavar='N', help='number of epochs to save the model')
    parser.add_argument('--pretrainedPath', default='weights/', type=str, metavar='DIR', help='path to pretrained model weight')
    parser.add_argument('--use_pretrained', action='store_true', default = False)
    args = parser.parse_args()

    #this is the setting for data augmentation
    transform = {}
    transform['random_horizontal_flips'] = 0.5
    transform['random_rotation'] = [-20, 20]
    transform['input_width'] = args.input_size
    transform['input_height'] = args.input_size
    transform['means_stds'] = args.means_stds

    #LOADING THE DATASET
    trainLoader, valLoader, class_name = getLoader(args.dataset_root, transform, args.batch_size)
    trainDatasetSize = len(trainLoader.dataset)
    print('train dataset len: {}'.format(trainDatasetSize))
    valDatasetSize = len(valLoader.dataset)
    print('validation dataset len: {}'.format(valDatasetSize))
    #BUILDING THE NETWORK
    print('Building {} network'.format(args.model_type))
    if not args.use_pretrained:
        args.pretrainedPath = None
    cnn_model = get_model(args.model_type, len(class_name), args.input_size, args.pretrainedPath)
    print('Finish building the network')
    criterion = nn.CrossEntropyLoss()#build loss criterion
    optimizer_cnn_model = optim.Adam(cnn_model.parameters(), args.lr)#build training optimizer
    step_size = int(args.epochs / 3)
    lr_train_scheduler = lr_scheduler.StepLR(optimizer_cnn_model, step_size=step_size, gamma=0.1)#build learning scheduler

    best_epoch = 0
    best_acc_val = 0.0
    start_epoch = 0
    n_iter = 0
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = os.path.join(args.checkpoint_dir, current_time)

    if args.resume:
        checkpoints_dir = args.checkpoint_dir
        train_checkpoints_path = os.path.join(checkpoints_dir,'training_checkpoint.pth.tar')
        checkpoint = torch.load(train_checkpoints_path)
        start_epoch = checkpoint['epoch']
        n_iter = checkpoint['n_iter']
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint['best_epoch']
        best_acc_val = checkpoint['best_acc_val']
        print('Resuming training from epoch {}'.format(start_epoch))
    else:
        try:
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)
        except os.error:
            print('Can\'t create checkpoint folder for the experiment')
            quit()
    train_checkpoints_path = os.path.join(checkpoints_dir,'training_checkpoint.pth.tar')
    classes_name_path = os.path.join(checkpoints_dir, 'classes_name.txt')
    write_class_name(classes_name_path, class_name)
    writer = SummaryWriter(log_dir=checkpoints_dir)#create tensorboard logging file 
    for param in cnn_model.parameters():#enable all cnn model training parameter
        param.requires_grad = True
    if args.use_gpu:#load the model and the criterion in the GPU
        cnn_model.cuda()
    cnn_model.train()#set cnn_model on the train mode

    for epoch in range(start_epoch, args.epochs):
        # train the network
        running_loss = 0.0
        running_corrects = 0
        for i, (img, label) in enumerate(trainLoader):
            #load all the data in GPU
            if args.use_gpu:
                img = img.cuda()
                label = label.cuda()
            #change the data type
            img = torch.autograd.Variable(img)
            label = torch.autograd.Variable(label)
            #set gradient to zero
            optimizer_cnn_model.zero_grad()
            #inference the input
            outputs = cnn_model(img)
            #get the training prediction
            _, preds = torch.max(outputs, 1)
            #compute the loss
            loss = criterion(outputs, label)
            #compute the gradient
            loss.backward()
            #update the model
            optimizer_cnn_model.step()
            lr_train_scheduler.step()
            running_loss += loss.item() * img.size(0)
            running_corrects += torch.sum(preds == label.data)
            writer.add_scalar('Loss_Logging/loss_iteration',loss.item(),n_iter)
            n_iter += 1
            if ((i % SHOW_LOG_EVERY_N_ITERATIONS) == 0):
                print('[Epoch {}/{}] Iteration: {}. Loss: {}'.format(epoch, args.epochs, i, loss.item()))
        epoch_loss = running_loss / trainDatasetSize
        epoch_acc = running_corrects.double() / trainDatasetSize
        print('[Epoch {}/{}] Loss: {:.4f}. Training Acc: {:.4f}'.format(epoch, args.epochs, epoch_loss, epoch_acc))
        writer.add_scalar('Loss_Logging/loss_epoch', epoch_loss, epoch)    
        writer.add_scalar('Accuracy/train_set', epoch_acc, epoch)    

        #save checkpoint, then validate the network
        if ((epoch % args.save_freq) == 0):
            best_acc_val, best_epoch = Saving_Checkpoint(epoch, n_iter, best_epoch, best_acc_val, args, cnn_model, 
                valLoader, valDatasetSize, checkpoints_dir, train_checkpoints_path, writer)

    best_acc_val, best_epoch = Saving_Checkpoint(epoch, n_iter, best_epoch, best_acc_val, args, cnn_model, 
        valLoader, valDatasetSize, checkpoints_dir, train_checkpoints_path, writer)

    the_text = 'model_type: {}, input_size: {}, means_stds:{}, lr: {}, batch_size: {}, \n'.format(
        args.model_type, args.input_size, args.means_stds, args.lr, args.batch_size) 
    the_text += 'The best Accuracy is {} at epoch {}'.format(best_acc_val, best_epoch)
    the_text_path = os.path.join(checkpoints_dir,'train_results.txt')
    the_file = open(the_text_path, 'w')
    the_file.write(the_text)
    the_file.close()
    print(the_text)

if __name__ == '__main__':
    run()
