import cv2
import json
import onnxruntime
import numpy as np

from .dnn import dnn

class ImgClassifier(dnn):
    def __init__(self, cfg_json_path, weight_path=None):
        f = open(cfg_json_path)
        data = json.load(f)
        if weight_path is not None:
            self.ort_session = onnxruntime.InferenceSession(weight_path, None, providers=["CUDAExecutionProvider"])
        else:
            self.ort_session = onnxruntime.InferenceSession(data['weights_path'], None, providers=["CUDAExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name
        self.class_name = data['class_name']
        dnn.__init__(self, data['input_size'], data['means'], data['stds'])

    def forward(self, image):
        batch = self.preprocess(image)
        idx = np.argmax(self.ort_session.run(None, {self.input_name: batch})[0])
        return self.class_name[idx]

if __name__ == "__main__":
    cfg_json_path = 'E:/Albert_Christianto/Project/defect_detection/app/cfgs/ResNet50_ImgClassifier.json'
    weight_path = 'E:/Albert_Christianto/Project/defect_detection/app/weights/ResNet50_ImgClassifier.onnx'
    engine = ImgClassifier(cfg_json_path, weight_path)
    img_path = 'E:/Albert_Christianto/Project/defect_detection/app/samples/mt_normal.jpg'
    image = cv2.imread(img_path)
    output = engine.forward(image)
    print(output)
