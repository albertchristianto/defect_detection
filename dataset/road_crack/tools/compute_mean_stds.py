import os
import cv2
import numpy as np

dataset_root = '../'
folderList = os.listdir(dataset_root)

channels_sum, channels_squared_sum, num_batches = 0, 0, 0

mean_stds_path = os.path.join(dataset_root, 'mean_stds.txt')
mean_stds_Txt = open(mean_stds_path, 'w')
for folderName in folderList:
    class_folder_list_path = dataset_root + '/' + folderName
    if os.path.isfile(class_folder_list_path) or folderName == "tools":
        continue
    datalist_folder_path = class_folder_list_path + '/' + 'datalist'
    if not os.path.exists(datalist_folder_path):
        print("Datalist folder is not found !!!")
        quit()
    train_datalists_list = list(filter(lambda x: x.startswith('train_'), os.listdir(datalist_folder_path)))
    for train_datalist in train_datalists_list:
        each_datalist_path = os.path.join(datalist_folder_path, train_datalist)
        theFile = open(each_datalist_path)
        theFile = theFile.readlines()
        for theFileNow in theFile:
            imgPath = dataset_root + '/' + theFileNow.replace('\n','').split(' ')[0]
            img = cv2.imread(imgPath)
            img = img.astype(np.float32) / 255.0
            img = img[:, :, ::-1]# swap bgr to rgb
            channels_sum += np.mean(img,  axis=(0, 1))
            channels_squared_sum += np.mean(img**2,  axis=(0, 1))
            num_batches += 1

means = channels_sum / num_batches
stds = (channels_squared_sum / num_batches - means ** 2) ** 0.5
for each_mean in means:
    mean_stds_Txt.write('{},'.format(each_mean))
mean_stds_Txt.write('\n')
for each_std in stds:
    mean_stds_Txt.write('{},'.format(each_std))
mean_stds_Txt.close()