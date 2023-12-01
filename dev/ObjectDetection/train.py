import os 
import sys
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import load_dataset, ctDataset
from model import get_model
from loss import FocalLoss, RegL1Loss
from tensorboardX import SummaryWriter

import math
import argparse
from loguru import logger
from test import test

LOG_LEVEL = 'TRACE'
SHOW_LOG_EVERY_N_ITERATIONS = 5
USE_GPU = True
PRETRAINED_PATH = './weights/'

logger.remove()
logger.add(sys.stdout, level=LOG_LEVEL)

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def define_load_training_param():
    logger.trace("Define the training-testing parameter")
    parser = argparse.ArgumentParser(description='PyTorch Image Classification Training Code by Albert Christianto')
    parser.add_argument('--dataset_root', default='/media/dtv4070ti-1/WOOS/Dataset/Inventories', required=False, type=str, help='path to the dataset')
    parser.add_argument('--model_type', type=str, default='efficientnetv2_m', help='define the model type that will be used')
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=1.25e-4, type=float, help='initial learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--val_freq', default=1, type=int, help='')
    parser.add_argument('--checkpoint_dir', default='checkpoint', type=str, help='path to the checkpoint')
    parser.add_argument('--resume', action='store_true', default = False)
    # parser.add_argument('--mode', type=str, required=True, help='define the model type that will be used')
    parser.add_argument('--v1_weights_path', default=None, type=str, metavar='DIR', help='path to weight of the model')
    args = parser.parse_args()

    return args

def load_dataset_and_create_dataloader(args, cnn_model):
    train_data, val_data =load_dataset(args.dataset_root)
    #cnn model must be created here!!! --> to get the number of classes here !!!
    train_dataset = ctDataset(train_data, cnn_model.input_size, cnn_model.stride)
    val_dataset = ctDataset(val_data, cnn_model.input_size, cnn_model.stride, val_mode=False)
    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=None, num_workers=1, shuffle=True)
    valLoader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True)
    return trainLoader, valLoader

def create_model(args):
    logger.info(f'Building {args.model_type} network')
    cnn_model = get_model(args.model_type, 4, 'weights/')

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
    hm_criterion = FocalLoss()#build loss criterion
    # hm_criterion = RegL1Loss()#build loss criterion
    bboxes_criterion = RegL1Loss()
    offsets_criterion = RegL1Loss()
    # qty_criterion = RegL1Loss()
    optimizer_cnn_model = optim.Adam(cnn_model.parameters(), args.lr, weight_decay=1e-4)
    # optimizer_cnn_model = optim.SGD(cnn_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lf = one_cycle(1, 0.002, args.epochs)  # cosine 1->hyp['lrf']

    lr_train_scheduler = optim.lr_scheduler.LambdaLR(optimizer_cnn_model, lr_lambda=lf)
    return hm_criterion, bboxes_criterion, offsets_criterion, optimizer_cnn_model, lr_train_scheduler#, qty_criterion

def saving_checkpoint(device, epoch, n_iter, best_epoch, best_acc_val, args, cnn_model, valLoader, 
                       checkpoints_dir, train_checkpoints_path, writer):
    acc_data, _ = test(cnn_model, valLoader, device, checkpoints_dir)
    writer.add_scalar('acc/map', acc_data[3], epoch)
    if (acc_data[3] > best_acc_val):
        best_acc_val = acc_data[3]
        best_epoch = epoch
        model_save_filename = os.path.join(checkpoints_dir,'obj_detector_best_epoch.pth')
        torch.save(cnn_model.state_dict(), model_save_filename)
    logger.info(f'Saving checkpoint to {train_checkpoints_path}')
    torch.save({'epoch':epoch, 'n_iter':n_iter, 'last_lr': args.lr, 'best_epoch': best_epoch, 'best_acc_val': best_acc_val,
                'model_state_dict':cnn_model.state_dict()}, train_checkpoints_path)
    return best_acc_val, best_epoch
# working seed:
# -> 10803418289028732205
# -> 12428790035814731680
def train(args):
    accum_iter = 32
    args.batch_size = int(args.batch_size / accum_iter)
    cnn_model = create_model(args)
    trainLoader, valLoader = load_dataset_and_create_dataloader(args, cnn_model)

    best_epoch, best_acc_val, start_epoch, n_iter, args, cnn_model = resume_or_new_training(args, cnn_model)
    hm_criterion, bboxes_criterion, offsets_criterion, optimizer_cnn_model, lr_train_scheduler = create_loss_function_optimizer_lr_scheduler(args, cnn_model)
    train_checkpoints_path = os.path.join(args.checkpoint_dir,'training_checkpoint.pth.tar')
    writer = SummaryWriter(log_dir=args.checkpoint_dir)
    use_gpu = USE_GPU and torch.cuda.is_available()
    device = 'cpu'
    if use_gpu:
        cnn_model.cuda()
        device = 'cuda:0'
    if args.v1_weights_path is not None:
        cnn_model.load_weights_v1(args.v1_weights_path)
    cnn_model.train()

    qty_weight = 0.0
    if start_epoch==0:
        torch.save({'epoch':0, 'n_iter':n_iter, 'last_lr': args.lr, 'best_epoch': best_epoch, 'best_acc_val': best_acc_val,
                    'model_state_dict':cnn_model.state_dict()}, train_checkpoints_path)
    for epoch in range(start_epoch, args.epochs):
        optimizer_cnn_model.zero_grad()
        total_loss = 0.0
        batch_loss = 0.0
        for i, [img, hm, wh, offset, reg_mask, qty, qty_mask] in enumerate(trainLoader):
            #load all the data in GPU
            if use_gpu:
                img = img.cuda()
                hm = hm.cuda()
                wh = wh.cuda()
                offset = offset.cuda()
                reg_mask = reg_mask.cuda()
                # qty = qty.cuda()
                # qty_mask = qty_mask.cuda()
            outputs = cnn_model(img)#inference the input
            # pred_hm = _sigmoid(outputs[0])
            pred_hm = outputs[0]
            pred_wh = outputs[1]
            pred_offset = outputs[2]
            hm_loss = hm_criterion(pred_hm, hm)
            # hm_mask = torch.ones_like(hm)
            # hm_loss = hm_criterion(pred_hm, hm, hm_mask)
            wh_loss = bboxes_criterion(pred_wh, wh, reg_mask)
            off_loss = offsets_criterion(pred_offset, offset, reg_mask)
            loss = 1.0 * hm_loss + 0.1 * wh_loss + 0.1 * off_loss 

            # Add gradients from this batch to saved ones, divide loss by NUM_ACCUMULATES if it's averaged over samples
            loss = loss / accum_iter 
            batch_loss += loss
            loss.backward()
            if ((i + 1) % accum_iter == 0) or (i + 1 == len(trainLoader)):
                writer.add_scalar('loss/train', batch_loss, n_iter)
                total_loss += batch_loss
                
                optimizer_cnn_model.step()# Update parameters using saved gradients
                # Zero saved gradients
                optimizer_cnn_model.zero_grad()
                
                if ((n_iter % SHOW_LOG_EVERY_N_ITERATIONS) == 0):
                    logger.info(f'Epoch [{epoch+1}/{args.epochs}], Iter [{i+1}/{len(trainLoader)}] Loss: {batch_loss}, average_loss: {total_loss*accum_iter/(i+1)}')
                batch_loss = 0
                n_iter += 1

        if ((epoch + 1) % args.val_freq) != 0:
            continue
        args.lr = optimizer_cnn_model.param_groups[0]['lr']
        # saving_checkpoint_tmp(epoch, n_iter, best_epoch, best_loss_val, args, cnn_model, train_loss, args.checkpoint_dir, train_checkpoints_path)
        best_acc_val, best_epoch= saving_checkpoint(device, epoch, n_iter, best_epoch, best_acc_val, args, cnn_model, valLoader, 
                       args.checkpoint_dir, train_checkpoints_path, writer)
        lr_train_scheduler.step()
        if best_acc_val == 0:
            logger.error("Failed training!!")
            quit()

if __name__ == "__main__":
    args = define_load_training_param()
    train(args)
