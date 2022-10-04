#----------------------------------------------------------------------
#register your model here
from models.VGG16 import Get_VGG16
from models.ResNet34 import Get_ResNet34
from models.ResNet50 import Get_ResNet50
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#function for calling the image classification model
#----------------------------------------------------------------------
def get_model(modelType, num_output, inp_size, pretrainedPath=None):
    if modelType == 'VGG16':
        model = Get_VGG16(num_output, inp_size, pretrainedPath)
    elif modelType == 'ResNet34':
        model = Get_ResNet34(num_output, pretrainedPath)
    elif modelType == 'ResNet50':
        model = Get_ResNet50(num_output, pretrainedPath)
    else:
        print('Please check model type!\nStopping...')
        quit()
    return model
