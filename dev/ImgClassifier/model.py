#----------------------------------------------------------------------
import sys
sys.path.append("../")
import torch
from torch import Tensor
import torch.nn as nn
#register your model here
from models.backbone.efficientnet import Get_EfficientNet_Backbone
from models.backbone.resnet import Get_ResNet
from models.backbone.vgg import Get_VGG
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#function for calling the image classification model
#----------------------------------------------------------------------

class ImgClassifier(nn.Module):
    def __init__(self, backbone, num_classes, pretrained_path='weights/'):
        super(ImgClassifier, self).__init__()
        self.backbone, self.input_size = self.create_backbone(backbone, pretrained_path)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(self.backbone.out_features_size[2], self.backbone.out_features_size[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.backbone.out_features_size[2], self.backbone.out_features_size[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.backbone.out_features_size[2], num_classes)
        )
        # self.fc = nn.Linear(self.backbone.out_features_size[2], num_classes)
        self._initialize_weights_norm()

    def create_backbone(self, backbone, pretrained_path):
        if backbone.find('efficientnet') != -1:
            backbone = backbone.replace('efficientnet', '')
            return Get_EfficientNet_Backbone(backbone, pretrained_path)
        elif backbone.find('resnet') != -1:
            backbone = backbone.replace('resnet', '')
            return Get_ResNet(backbone, pretrained_path), 224
        elif backbone.find('vgg') != -1:
            backbone = backbone.replace('vgg', '')
            return Get_VGG(backbone, pretrained_path), 224
        else:
            return None, None

    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.backbone(x)
        x = self.avgpool(x[2])
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ =="__main__":
    model = ImgClassifier("efficientnetb3", 4)
    print(model)
    fake_input = torch.rand(1, 3, model.input_size, model.input_size, requires_grad=True)
    fake_output = model(fake_input)
    