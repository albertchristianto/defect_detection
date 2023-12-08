#----------------------------------------------------------------------
import sys
sys.path.append("..")
import torch
from torch import Tensor
import torch.nn as nn
#register your model here
from models.backbone.efficientnet import Get_EfficientNet_Backbone
from models.backbone.resnet import Get_ResNet
from models.backbone.vgg import Get_VGG
from models.neck.mdf_bifpn import MDF_BiFPN
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#function for calling the image classification model
#----------------------------------------------------------------------
ning_param = {
    'efficientnetv2_s': [386, 5, 256],
    'efficientnetv2_m': [386, 7, 256],
    'efficientnetv2_l': [512, 7, 256]
}

class SSA_Classifier(nn.Module):
    def __init__(self, backbone, neck_channels, neck_repeat_block, head_channels, s_attention=False, pretrained_path='weights/'):
        super(SSA_Classifier, self).__init__()
        self.backbone, self.input_size = self.create_backbone(backbone, pretrained_path)
        self.neck = MDF_BiFPN(self.backbone.out_features_size, feature_size=neck_channels, num_layers=neck_repeat_block, out_feature_size=neck_channels)
        self.classes_head = nn.Sequential(
                              nn.Conv2d(neck_channels, head_channels, kernel_size=3, padding=1, bias=True),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(head_channels, 2, kernel_size=1, stride=1, padding=0, bias=True)
                            )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(neck_channels, head_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(head_channels, head_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(head_channels, 2)
        )
        self.s_attention = s_attention
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
        out = self.backbone(x)
        out = self.neck(out)
        out_seg = self.classes_head(out)
        if self.s_attention:
            # print(out_seg.size())
            out = torch.concat([out, out_seg])
        out_classes = self.avgpool(out)
        out_classes = torch.flatten(out_classes, 1)
        out_classes = self.fc(out_classes)

        return out_classes, out_seg

if __name__ =="__main__":
    model_type = 'efficientnetv2_s'
    model = SSA_Classifier(model_type, ning_param[model_type][0], ning_param[model_type][1], ning_param[model_type][2])
    fake_input = torch.rand(1, 3, model.input_size, model.input_size, requires_grad=True)
    fake_output, fake_output_seg = model(fake_input)
    print(fake_output.size())