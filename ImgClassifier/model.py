#----------------------------------------------------------------------
#register your model here
from models.vgg import Get_VGG
from models.resnet import Get_ResNet18, Get_ResNet34, Get_ResNet50
from models.efficientnet import Get_EfficientNetB0
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#function for calling the image classification model
#----------------------------------------------------------------------
def get_model(modelType, num_output, inp_size, pretrainedPath=None):
    if modelType.find('VGG') != -1:
        model = Get_VGG(modelType, 3, num_output, inp_size, pretrainedPath)
    elif modelType == 'ResNet18':
        model = Get_ResNet18(num_output, pretrainedPath)
    elif modelType == 'ResNet34':
        model = Get_ResNet34(num_output, pretrainedPath)
    elif modelType == 'ResNet50':
        model = Get_ResNet50(num_output, pretrainedPath)
    elif modelType == 'EfficientNetB0':
        model = Get_EfficientNetB0(num_output, pretrainedPath)
    else:
        print('Please check model type!\nStopping...')
        quit()
    return model
