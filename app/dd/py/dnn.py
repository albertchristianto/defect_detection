import cv2
import numpy as np

class dnn:
    def __init__(self, input_size, means, stds):
        self.input_size = input_size
        self.means = means
        self.stds = stds

    def forward(self):
        raise NotImplementedError

    def preprocess(self, image):
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32) / 255.0
        image = image[:, :, ::-1]# swap bgr to rgb
        for i in range(3):
            image[:, :, i] = image[:, :, i] - self.means[i]
            image[:, :, i] = image[:, :, i] / self.stds[i]
        image = image.transpose((2, 0, 1)).astype(np.float32)
        batch_images= np.expand_dims(image, 0)
        return batch_images