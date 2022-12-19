import onnx
import onnxruntime

from dnn import dnn

class ImgClassifier(dnn):
    def __init__(self,  input_size, means, stds, output_weight_path):
        self.ort_session = onnxruntime.InferenceSession(output_weight_path, None, providers=["CUDAExecutionProvider"])
        dnn.__init__(self, input_size, means, stds)
    def forward(self, image):
        ort_inputs = {self.ort_session.get_inputs()[0].name: image}
        ort_outs = self.ort_session.run(None, ort_inputs)

if __name__ == "__main__":
    engine = ImgClassifier()
