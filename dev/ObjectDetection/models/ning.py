import torch
import torch.nn as nn
from models.backbone_efficientnet import Get_EfficientNet_Backbone
from models.backbone_resnet import Get_ResNet
from models.neck_bifpn import BiFPN

def fill_fc_weights(layers):
    for m in layers.modules():
        if not isinstance(m, nn.Conv2d):
            continue
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

ning_param = {
    'efficientnetv2_s': [386, 5, 256],
    'efficientnetv2_m': [386, 7, 256],
    'efficientnetv2_l': [512, 7, 256]
}

class Ning(nn.Module):
    def __init__(self, backbone, neck_channels, neck_repeat_block, head_channels, num_classes, pretrained_path='weights/', conf_thresh=0.3):
        super(Ning, self).__init__()
        self.stride = 4
        self.backbone, self.input_size = self.create_backbone(backbone, pretrained_path)
        # print(self.backbone.out_features_size)
        assert self.backbone is not None
        self.neck = BiFPN(self.backbone.out_features_size, feature_size=neck_channels, num_layers=neck_repeat_block, out_feature_size=neck_channels)
        self.classes_head = nn.Sequential(
                              nn.Conv2d(neck_channels, head_channels, kernel_size=3, padding=1, bias=True),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(head_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                            )
        self.classes_head[-1].bias.data.fill_(-2.19)
        self.bboxes_head = nn.Sequential(
                              nn.Conv2d(neck_channels, head_channels, kernel_size=3, padding=1, bias=True),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(head_channels, 2, kernel_size=1, stride=1, padding=0, bias=True)
                            )
        fill_fc_weights(self.bboxes_head)
        self.offsets_head = nn.Sequential(
                              nn.Conv2d(neck_channels, head_channels, kernel_size=3, padding=1, bias=True),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(head_channels, 2, kernel_size=1, stride=1, padding=0, bias=True)
                            )
        fill_fc_weights(self.offsets_head)
        self.conf_thresh = conf_thresh
        self.nms_head = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def _nms(self, heat):
        # shape = heat.size()
        # heat = heat.view(shape[0], 1, shape[1], shape[2])
        hmax = self.nms_head(heat)
        keep = (hmax == heat).float()
        thresh = (heat > self.conf_thresh).float()
        return heat * keep * thresh

    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path), strict=False)

    def forward(self, x):
        out = self.backbone(x)
        out = self.neck(out)
        out_classes = self.classes_head(out)
        out_bboxes = self.bboxes_head(out)
        out_offsets = self.offsets_head(out)
        if self.training:
            return out_classes, out_bboxes, out_offsets
        # return out_classes, out_bboxes, out_offsets, out_qty
        out_classes = torch.clamp(out_classes.sigmoid_(), min=1e-4, max=1-1e-4)
        out_classes = self._nms(out_classes)
        out_classes_scores, out_classes_idx = torch.max(out_classes, 1)
        return out_classes_scores, out_classes_idx, out_bboxes, out_offsets

    def create_backbone(self, backbone, pretrained_path):
        if backbone.find('efficientnet') != -1:
            backbone = backbone.replace('efficientnet', '')
            return Get_EfficientNet_Backbone(backbone, pretrained_path)
        elif backbone.find('resnet') != -1:
            backbone = backbone.replace('resnet', '')
            return Get_ResNet(backbone, pretrained_path), 512
        else:
            return None, None

if __name__ == '__main__':
    model_type = 'efficientnetv2_l'
    model = Ning(model_type, ning_param[model_type][0], ning_param[model_type][1], ning_param[model_type][2], 2)
    fake_input = torch.rand(1, 3, model.input_size, model.input_size, requires_grad=True)
    fake_output = model(fake_input)
    print(fake_output[0].size(), fake_output[1].size(), fake_output[2].size())