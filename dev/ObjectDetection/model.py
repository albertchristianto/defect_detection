# from SavvyIMS.Dev.CenterNet.ref.ning import Ning, ning_param
from models.ning import Ning, ning_param


def get_model(model_type, num_classes, pretrained_path='weights/', conf_thresh=0.3):
  model = Ning(model_type, ning_param[model_type][0], ning_param[model_type][1], ning_param[model_type][2], num_classes, pretrained_path=pretrained_path, conf_thresh=conf_thresh)
  return model
