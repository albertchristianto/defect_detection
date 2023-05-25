#----------------------------------------------------------------------
#register your model here
import torchvision.models.segmentation
import torch
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#function for calling the image classification model
#----------------------------------------------------------------------
def get_model(modelType, num_output, pretrained=True):
    if modelType == 'deeplabv3_mobilenet_v3':
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=pretrained)
        model.classifier[4] = torch.nn.Conv2d(256, num_output, kernel_size=(1, 1), stride=(1, 1))
    else:
        print('Please check model type!\nStopping...')
        quit()
    return model

if __name__ == '__main__':
    fake_input = torch.rand(2, 3, 224, 224, requires_grad=True)
    cnn_model = get_model('deeplabv3_mobilenet_v3', 2, False)
    output = cnn_model(fake_input)['out']
    print(output.size())