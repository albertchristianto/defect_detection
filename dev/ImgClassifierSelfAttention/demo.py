import cv2
from model import get_model
from dataloader import vgg_preprocess
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    modelPath = 'checkpoint/20230526_1004/img_segmentation_best_epoch.pth'
    img_path = 'samples/exp1_num_31917.jpg'
    img = cv2.imread(img_path)
    disp = img.copy()
    cnn_model = get_model('deeplabv3_mobilenet_v3', 2, False)
    cnn_model.load_state_dict(torch.load(modelPath)) # Load trained model
    cnn_model.eval()
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    means =  np.array([ 0.44145817, 0.44145817, 0.44145817 ])
    stds = np.array([ 0.17827073, 0.17827073, 0.17827073])
    img = vgg_preprocess(img, means, stds)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)

    output = cnn_model(img)['out'] 
    seg = torch.argmax(output, 1).cpu().detach().numpy()  # Get  prediction classes
    plt.imshow(disp[:,:,::-1])  # Show image
    plt.show()
    plt.imshow(seg[0])  # display image
    plt.show()
