import torch
import argparse

from model import get_model

import onnx
import onnxruntime
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description='LPR-Net 2 LibTorch. Made by Albert Christianto')
parser.add_argument('--weight_path', default='checkpoint/20220808-1234/epoch_90.pth', type=str, metavar='DIR',
                                        help='path to weight of the model')   
parser.add_argument('--model_type', type=str, default='ResNet34', help='define the model type that will be used: VGG16,')
parser.add_argument('--class_name_path', default='checkpoint/20220808-1234/classes_name.txt', type=str, metavar='DIR',
                                        help='path to weight of the model')
parser.add_argument('--input_size', default=160, type=int, metavar='N',
                                        help='number of epochs to save the model')
parser.add_argument('--output_weight_path', default='img_classifier.onnx', type=str, metavar='DIR',
                                        help='path to weight of the model')   
parser.add_argument('--opset_version', default=10, type=int, metavar='N',
                                        help='number of epochs to save the model')
args = parser.parse_args()
classes_name = get_class_name(args.class_name_path)
model = get_model(args.model_type, len(classes_name), args.input_size, None)
model.load_state_dict(torch.load(args.weight_path))
print(model)
fake_input = torch.rand(1, 3, args.input_size, args.input_size, requires_grad=True)
model.eval()
fake_output = model(fake_input)
# export the model into ONNX framework
torch.onnx.export(model,             # model being run
                  fake_input,        # model input (or a tuple for multiple inputs)
                  args.output_weight_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=args.opset_version,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']#, # the model's output names
                  #dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #              'output' : {0 : 'batch_size'}}
                  )
print('testing the conversion')

onnx_model = onnx.load(args.output_weight_path)
onnx.checker.check_model(onnx_model)

if torch.cuda.is_available():
    ort_session = onnxruntime.InferenceSession(args.output_weight_path, None, providers=["CUDAExecutionProvider"])
else:
    ort_session = onnxruntime.InferenceSession(args.output_weight_path, None)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(fake_input)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(fake_output), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
