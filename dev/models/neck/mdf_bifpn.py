import torch

from torch import nn
import torch.nn.functional as F

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv share bias between depthwise_conv and pointwise_conv or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, bias=False, groups=in_channels, padding=1)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True, groups=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = nn.SiLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        self.p3_td = SeparableConvBlock(feature_size, feature_size, activation=True)
        self.p4_td = SeparableConvBlock(feature_size, feature_size, activation=True)
        self.p5_td = SeparableConvBlock(feature_size, feature_size, activation=True)
        self.p6_td = SeparableConvBlock(feature_size, feature_size, activation=True)

        self.p4_out = SeparableConvBlock(feature_size, feature_size, activation=True)
        self.p5_out = SeparableConvBlock(feature_size, feature_size, activation=True)
        self.p6_out = SeparableConvBlock(feature_size, feature_size, activation=True)
        self.p7_out = SeparableConvBlock(feature_size, feature_size, activation=True)

        # TODO: Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 4))
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, 4))
        self.w2_relu = nn.ReLU()

    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x, p7_x = inputs
        
        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 = w1/(torch.sum(w1, dim=0) + self.epsilon)
        w2 = self.w2_relu(self.w2)
        w2 = w2/(torch.sum(w2, dim=0) + self.epsilon)
        
        p7_td = p7_x
        p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * F.interpolate(p7_td, scale_factor=2))
        p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * F.interpolate(p4_td, scale_factor=2))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * nn.Upsample(scale_factor=0.5)(p4_out))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p5_out))
        p7_out = self.p7_out(w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * nn.Upsample(scale_factor=0.5)(p6_out))

        return [p3_out, p4_out, p5_out, p6_out, p7_out]

class MDF_BiFPN(nn.Module):
    def __init__(self, size, feature_size=64, num_layers=5, out_feature_size=256):
        super(MDF_BiFPN, self).__init__()

        self.p3 = nn.Conv2d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p4 = nn.Conv2d(size[0], feature_size, kernel_size=1, stride=1, padding=0)
        self.p5 = nn.Conv2d(size[1], feature_size, kernel_size=1, stride=1, padding=0)
        self.p6 = nn.Conv2d(size[2], feature_size, kernel_size=1, stride=1, padding=0)

        # p7 is obtained via a 3x3 stride-2 conv on C5
        self.p7 = nn.Conv2d(size[2], feature_size, kernel_size=3, stride=2, padding=1)
        self.reduction = nn.Conv2d(feature_size*5, out_feature_size, kernel_size=1, stride=1, padding=0)
        bifpns = []
        for _ in range(num_layers):
            bifpns.append(BiFPNBlock(feature_size))
        self.bifpn = nn.Sequential(*bifpns)
        self.scale_factor = [0, 2, 4, 8, 16]
        self._initialize_weights_norm()

    def forward(self, inputs):
        c3, c4, c5 = inputs

        # Calculate the input column of BiFPN
        p3_x = self.p3(F.interpolate(c3, scale_factor=2))
        p4_x = self.p4(c3)
        p5_x = self.p5(c4)
        p6_x = self.p6(c5)
        p7_x = self.p7(c5)

        features = [p3_x, p4_x, p5_x, p6_x, p7_x]
        features = self.bifpn(features)
        for i in range(1, len(features)):
            features[i] = nn.Upsample(scale_factor=self.scale_factor[i], mode='nearest')(features[i])
        features = torch.cat(features, axis=1)
        features = self.reduction(features)
        return features

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

if __name__ == '__main__':
    input_size = 60
    features_size = [80, 176, 1280]
    model = MDF_BiFPN(features_size, feature_size=112)
    input_size = 64
    fake_input1 = torch.rand(1, features_size[0], input_size, input_size, requires_grad=True)
    fake_input2 = torch.rand(1, features_size[1], input_size//2, input_size//2, requires_grad=True)
    fake_input3 = torch.rand(1, features_size[2], input_size//4, input_size//4, requires_grad=True)
    fake_output = model([fake_input1, fake_input2, fake_input3])
    print(fake_output.size())
