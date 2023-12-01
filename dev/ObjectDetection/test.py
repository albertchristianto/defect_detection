import argparse
import json
import cv2
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import box_iou
from utils.metrics import ap_per_class, ConfusionMatrix

from dataset import reverse_res_to_bbox



import sys
from loguru import logger
from torch.utils.data import DataLoader
from dataset import load_dataset, ctDataset
from model import get_model_v2

def qty_acc_processing(qty, qty_mask, qty_out):
    correct = 0
    out_filter = torch.nonzero(qty_mask)
    for each in out_filter:
        gt = int(qty[each[0],each[1],each[2]]*100.0)
        pred = int(qty_out[each[0],each[1],each[2]]*100.0)
        correct += int(gt == pred)
    return correct, out_filter.size()[0]

def bbox_post_processing(scores, classes_idx, wh, off, stride=4, det_scale=1.0):
    ycxc = torch.nonzero(scores)

    #must be process in
    new_scores = scores[ycxc[:,0], ycxc[:,1]]
    new_cls = classes_idx[ycxc[:,0], ycxc[:,1]]
    new_wh = wh[:, ycxc[:,0], ycxc[:,1]]
    new_off = off[:, ycxc[:,0], ycxc[:,1]]

    # print(ycxc.size(), new_wh.size(), new_cls.size(), new_off.size(), new_scores.size())
    # print(xcyc.size())
    # qty_out = torch.round(qty[:, ycxc[:,0], ycxc[:,1]] * (new_cls == 2) * 100.0)
    # print(qty_out.size())

    bboxes = []
    x1 = (ycxc[:,1] - (new_wh[1,:]/2.0) + new_off[1,:]) * stride / det_scale
    y1 = (ycxc[:,0] - (new_wh[0,:]/2.0) + new_off[0,:]) * stride / det_scale
    x2 = (ycxc[:,1] + (new_wh[1,:]/2.0) + new_off[1,:]) * stride / det_scale
    y2 = (ycxc[:,0] + (new_wh[0,:]/2.0) + new_off[0,:]) * stride / det_scale
    bboxes = torch.stack((x1, y1, x2, y2, new_scores, new_cls), 1)

    # print(new_cls.size(), x1.size(), y1.size(), x2.size(), y2.size())
    # print(bboxes.size())
    return bboxes#, qty_out[0,:] #.cpu().detach().numpy()

LOG_LEVEL = 'TRACE'
logger.remove()
logger.add(sys.stdout, level=LOG_LEVEL)

def test(model, dataloader, device, save_dir, training=False, plots= True):
    model.eval()

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    names = {0: "product", 1: "product-label", 2: "compartment", 3: "compartment-label"}
    nc=4
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []
    for batch_i, [img, hm, wh, offset, reg_mask, qty, qty_mask] in enumerate(tqdm(dataloader, desc=s)):#, disp] in enumerate(tqdm(dataloader, desc=s)):
        #load all the data in target device
        img = img.to(device)
        hm = hm.to(device)
        wh = wh.to(device)
        offset = offset.to(device)
        reg_mask = reg_mask.to(device)
        # qty = qty.to(device)
        # qty_mask = qty_mask.to(device)

        # out_classes_scores, out_classes_idx, out_bboxes, out_offsets, out_qty = None, None, None, None, None
        with torch.no_grad():
            out_classes_scores, out_classes_idx, out_bboxes, out_offsets = model(img)#inference the input
        # Statistics per image
        for si, pred in enumerate(hm):#iterating for each image in batch
            # disp_now = disp[si, :, :, :].detach().numpy()
            labels = reverse_res_to_bbox(hm[si, :, :, :], wh[si, :, :, :], offset[si, :, :, :])
            # for each_label in labels:
            #     # print(each_label)
            #     disp_now = cv2.rectangle(disp_now, (int(each_label[1]), int(each_label[2])), (int(each_label[3]), int(each_label[4])), (50, 10, 70), 2)
            # cv2.imshow("test", disp_now)
            # cv2.waitKey(0)
            # print("label:", labels)
            pred = bbox_post_processing(out_classes_scores[si, :, :], out_classes_idx[si, :, :], 
                                     out_bboxes[si, :, :, :], out_offsets[si, :, :, :])
            # print("pred:", pred)
            # labels = targets[targets[:, 0] == si, 1:]
            # correct, n_data = qty_acc_processing(qty[si, :, :, :], qty_mask[si, :, :, :], out_qty[si, :, :, :])
            # total_correct_qty += correct
            # total_qty_data += n_data
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            # scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = labels[:, 1:5]
                # scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if ((nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Return results
    model.train()  # for training

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps

def define_load_param():
    logger.trace("Define the training-testing parameter")
    parser = argparse.ArgumentParser(description='PyTorch Image Classification Training Code by Albert Christianto')
    parser.add_argument('--dataset_root', default='/media/dtv4070ti-1/WOOS/Dataset/Inventories', required=False, type=str, help='path to the dataset')
    parser.add_argument('--model_type', type=str, default='efficientnetv2_m', help='define the model type that will be used')
    parser.add_argument('--weight_path', default="checkpoint/20230816_1547/obj_detector_best_epoch.pth", type=str, metavar='DIR', help='path to weight of the model')
    parser.add_argument('--batch_size', type=int, default=2, help='size of each image batch')
    args = parser.parse_args()

    return args

def load_dataset_and_create_dataloader(args, input_size, stride, batch_size):
    _, val_data = load_dataset(args.dataset_root)
    val_dataset = ctDataset(val_data, input_size, stride, val_mode=True)
    valLoader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    return valLoader

if __name__ == '__main__':
    args = define_load_param()
    logger.info(f'Building {args.model_type} network')
    model = get_model_v2(args.model_type, 4, conf_thresh=0.3)
    model.load_state_dict(torch.load(args.weight_path))
    valLoader = load_dataset_and_create_dataloader(args, model.input_size, model.stride, args.batch_size)
    save_dir = os.path.dirname(args.weight_path)
    device = 'cuda:0'
    model.to(device)
    acc_data, maps, _ = test(model, valLoader, device, save_dir)
