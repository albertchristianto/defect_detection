import cv2
import onnxruntime
import numpy as np

from dnn import dnn

class ImgClassifier(dnn):
    def __init__(self,  input_size, class_name, means, stds, weight_path):
        self.ort_session = onnxruntime.InferenceSession(weight_path, None, providers=["CUDAExecutionProvider"])
        self.input_name = self.ort_session.get_inputs()[0].name
        self.class_name = class_name
        dnn.__init__(self, input_size, means, stds)
    def forward(self, image):
        batch = self.preprocess(image)
        idx = np.argmax(self.ort_session.run(None, {self.input_name: batch})[0])
        return self.class_name[idx]

if __name__ == "__main__":
    cfg_json_path = 'E:/Albert_Christianto/Project/defect_detection/app/cfgs/ResNet50_ImgClassifier.json'
    weight_path = 'E:/Albert_Christianto/Project/defect_detection/app/weights/img_classifier.onnx'
    import json
    f = open(cfg_json_path)
    data = json.load(f)
    engine = ImgClassifier(data['input_size'], data['class_name'], data['means'], data['stds'], weight_path)
    img_path = 'E:/Albert_Christianto/Project/defect_detection/app/samples/mt_normal.jpg'
    image = cv2.imread(img_path)
    output = engine.forward(image)
    print(output)
