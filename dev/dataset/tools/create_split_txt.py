import os
import random

dataset_root = '../'
folderList = os.listdir(dataset_root)

classes_name_path = os.path.join(dataset_root, 'classes_name.txt')
classes_name_Txt = open(classes_name_path, 'w')
for folderName in folderList:
    if folderName == "tools":
        continue
    class_folder_list_path = dataset_root + '/' + folderName
    if os.path.isfile(class_folder_list_path):
        continue
    datalist_path = class_folder_list_path + '/' + 'datalist'
    if not os.path.exists(datalist_path):
        os.makedirs(datalist_path)
    class_folder_list = os.listdir(class_folder_list_path)
    for class_folder in class_folder_list:
        if class_folder == "datalist":
            continue
        rootPath = class_folder_list_path + '/' + class_folder
        trainPath = class_folder_list_path + '/datalist/train_'+ class_folder +'.txt'
        valPath = class_folder_list_path + '/datalist/val_'+ class_folder +'.txt'
        if os.path.exists(trainPath) and os.path.exists(valPath):
            print('train list txt and val list txt for ' + class_folder + ' is ready')
            continue

        print('Generating train list txt and val list txt for ' + class_folder)
        trainTxt = open(trainPath, 'w')
        valTxt = open(valPath, 'w')
        listImg = list(filter(lambda x: x.endswith('.jpg'), os.listdir(rootPath)))

        for i, itemNow in enumerate(listImg):
            if i % 100==0:
                print('Processed {} images'.format(i))
            filePath = folderName + '/' + class_folder + '/' + itemNow
            if random.random() < 0.8:
                trainTxt.write('{}\n'.format(filePath))
            else:
                valTxt.write('{}\n'.format(filePath))
        trainTxt.close()
        valTxt.close()
    classes_name_Txt.write('{}\n'.format(folderName))
classes_name_Txt.close()