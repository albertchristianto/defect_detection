import os
import torch

from model import get_model_v2

import onnx
import onnxruntime
import numpy as np
import argparse
from loguru import logger

def define_load_param():
    logger.trace("Define the training-testing parameter")
    parser = argparse.ArgumentParser(description='PyTorch Image Classification Training Code by Albert Christianto')
    parser.add_argument('--model_type', type=str, default='efficientnetv2_m', help='define the model type that will be used')
    parser.add_argument('--weight_path', default="checkpoint/20230721_1738/obj_detector_best_epoch.pth", type=str, metavar='DIR', help='path to weight of the model')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = define_load_param()
    model = get_model_v2(args.model_type, 4, infer_mode=True, conf_thresh=0.3)
    model.load_state_dict(torch.load(args.weight_path))
    model.eval()

    fake_input = torch.rand(1, 3, model.input_size, model.input_size, requires_grad=True)
    fake_out_classes_scores, fake_out_classes_idx, fake_out_bboxes, fake_out_offsets, fake_out_qty = model(fake_input)
    time_str = os.path.dirname(args.weight_path).split('/')[-1]
    onnx_path =  f'center_net_ning_v2_{time_str}.onnx'
    # export the model into ONNX framework
    torch.onnx.export(model,             # model being run
                    fake_input,        # model input (or a tuple for multiple inputs)
                    onnx_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output']  # the model's output names
                    #dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    #              'output' : {0 : 'batch_size'}}
                    )
    logger.info('testing the conversion')

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    if torch.cuda.is_available():
        logger.info('Using CUDA!')
        ort_session = onnxruntime.InferenceSession(onnx_path, None, providers=["CUDAExecutionProvider"])
    else:
        logger.info('Using CPU!')
        ort_session = onnxruntime.InferenceSession(onnx_path, None)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = { ort_session.get_inputs()[0].name: to_numpy(fake_input) }
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(fake_out_classes_scores), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(fake_out_classes_idx), ort_outs[1], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(fake_out_bboxes), ort_outs[2], rtol=1e-02, atol=1e-04)
    np.testing.assert_allclose(to_numpy(fake_out_offsets), ort_outs[3], rtol=1e-02, atol=1e-04)
    np.testing.assert_allclose(to_numpy(fake_out_qty), ort_outs[4], rtol=1e-02, atol=1e-04)

    logger.info("Exported model has been tested with ONNXRuntime, and the result looks good!")