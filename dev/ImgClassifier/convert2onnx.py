import os
import torch
import argparse

from model import ImgClassifier

import onnx
import onnxruntime
import numpy as np
from dataloader import get_class_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LPR-Net 2 LibTorch. Made by Albert Christianto')
    parser.add_argument('--exp_folder_path', default='checkpoint/20230908_1601_efficientnetb2', type=str, metavar='DIR',
                                            help='path to weight of the model')
    parser.add_argument('--model_type', type=str, default='efficientnetb2', help='define the model type that will be used: VGG16,')
    parser.add_argument('--opset_version', default=11, type=int, metavar='N',
                                            help='number of epochs to save the model')
    args = parser.parse_args()
    class_name_path = os.path.join(args.exp_folder_path, "classes_name.txt")
    classes_name = get_class_name(class_name_path)
    weights_path = os.path.join(args.exp_folder_path, "img_classifier_best_epoch.pth")
    output_weights_path = os.path.join(args.exp_folder_path, "img_classifier_best_epoch.onnx")

    model = ImgClassifier(args.model_type, len(classes_name))
    model.load_state_dict(torch.load(weights_path))
    print(model)
    fake_input = torch.rand(1, 3, model.input_size, model.input_size, requires_grad=True)
    model.eval()
    fake_output = model(fake_input)
    # export the model into ONNX framework
    torch.onnx.export(model,             # model being run
                    fake_input,        # model input (or a tuple for multiple inputs)
                    output_weights_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=args.opset_version,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output']#, # the model's output names
                    #dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    #              'output' : {0 : 'batch_size'}}
                    )
    print('testing the conversion')

    onnx_model = onnx.load(output_weights_path)
    onnx.checker.check_model(onnx_model)

    if torch.cuda.is_available():
        print('Using CUDA!')
        ort_session = onnxruntime.InferenceSession(output_weights_path, None, providers=["CUDAExecutionProvider"])
    else:
        print('Using CPU!')
        ort_session = onnxruntime.InferenceSession(output_weights_path, None)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = { ort_session.get_inputs()[0].name: to_numpy(fake_input) }
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(fake_output), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
