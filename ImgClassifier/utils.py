import os
import numpy as np

vgg_means = [0.5, 0.5, 0.5] #[0.485, 0.456, 0.406]
vgg_stds = [1.0, 1.0, 1.0] #[0.229, 0.224, 0.225]

def vgg_preprocess(image):
    image = image.astype(np.float32) / 255.0
    preprocessed_img = image.copy()[:, :, ::-1]# swap bgr to rgb
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - vgg_means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / vgg_stds[i]
    preprocessed_img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
    return preprocessed_img

def get_class_name(class_name_path):
    class_name_file = open(class_name_path)
    class_name = []
    class_name_file = class_name_file.readlines()
    for theFileNow in class_name_file:
        theFileNow = theFileNow.replace('\n','').split(' ')[0]
        class_name.append(theFileNow)
    return class_name

def write_class_name(classes_name_path, class_name):
    classes_name_Txt = open(classes_name_path, 'w')
    for class_name_now in class_name:
        classes_name_Txt.write('{}\n'.format(class_name_now))
    classes_name_Txt.close()

def loadtxtfiles(dataset_root):
    class_name_path = os.path.join(dataset_root, 'classes_name.txt')
    class_name = get_class_name(class_name_path)
    train_data = []
    val_data = []
    len_data = {}

    for class_idx, each_class in enumerate(class_name):
        len_data[each_class] = 0
        class_folder_datalist_path = dataset_root + '/' + each_class + '/datalist'
        class_datalist_list = os.listdir(class_folder_datalist_path)
        for each_datalist in class_datalist_list:
            each_datalist_path = class_folder_datalist_path + '/' + each_datalist
            theFile = open(each_datalist_path)
            theFile = theFile.readlines()
            for theFileNow in theFile:
                theFileNow = dataset_root + '/' + theFileNow.replace('\n','').split(' ')[0]
                each_data = []
                each_data.append(theFileNow)
                each_data.append(class_idx)
                if (each_datalist.startswith('train')):
                    len_data[each_class] += 1 #compute the how many class inside the training data per class
                    train_data.append(each_data)
                    continue
                if (each_datalist.startswith('val')):
                    val_data.append(each_data)
                    continue

    all_data = 0
    for each_key in len_data:
        all_data = all_data + len_data[each_key]
    weight_value_default = 1.0 / all_data
    weight_value = {}
    for each_key in len_data:
        weight_value[each_key] = weight_value_default / float(len_data[each_key])
    weight_samples = np.zeros(all_data)
    last_idx = 0
    for each_class in class_name:
        weight_samples[last_idx:(last_idx + len_data[each_class])] = weight_value[each_class]
        last_idx += len_data[each_class]

    return train_data, val_data, weight_samples, class_name