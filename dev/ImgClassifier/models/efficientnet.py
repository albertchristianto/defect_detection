import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from math import ceil

pre_trained_weights_url = {
    "b0":'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth',
    "b1": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
    "b2": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth',
    "b3": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b3_ra2-cf984f9c.pth',
    "b4": 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_320-7eb33cd5.pth',
    "b5": 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    "b6": 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    "b7": 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth'
}

base_model = [
    #base_model_expand_ratio, channels, layers, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

#read this https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ 
pre_defined_values = {
    #tuple of: (width, depth, resolution, drop_rate)
    "b0": (1.0, 1.0, 224, 0.2),
    "b1": (1.0, 1.1, 240, 0.2),
    "b2": (1.1, 1.2, 260, 0.3),
    "b3": (1.2, 1.4, 300, 0.3),
    "b4": (1.4, 1.8, 380, 0.4),
    "b5": (1.6, 2.2, 456, 0.4),
    "b6": (1.8, 2.6, 528, 0.5),
    "b7": (2.0, 3.1, 600, 0.5)
}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() # SiLU <-> Swish
        self._initialize_weights_norm()

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

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )
        self._initialize_weights_norm()

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

    def forward(self, x):
        return x * self.se(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio,
            reduction=4, # squeeze excitation
            survival_prob=0.8): # for stochastic depth
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock( in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Sequential(
            CNNBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self._initialize_weights_norm()

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

    def stochastic_depth(self, x):#this is the stochastic depth layer
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor 
        #the logic here should be {x * binary_tensor}, torch.div(x, self.survival_prob) is from the stochastic depth paper

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)

class CondConvResidual(InvertedResidualBlock):
    """ Inverted residual block w/ CondConv routing"""

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d, se_layer=None, num_experts=0, drop_path_rate=0.):

        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)

        super(CondConvResidual, self).__init__(
            in_chs, out_chs, dw_kernel_size=dw_kernel_size, stride=stride, dilation=dilation, group_size=group_size,
            pad_type=pad_type, act_layer=act_layer, noskip=noskip, exp_ratio=exp_ratio, exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size, se_layer=se_layer, norm_layer=norm_layer, conv_kwargs=conv_kwargs,
            drop_path_rate=drop_path_rate)

        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, x):
        shortcut = x
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

class EfficientNet(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNet, self).__init__()
        depth_factor, width_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )
        self._initialize_weights_norm()

    def calculate_factors(self, version):
        depth_factor, width_factor, _, dropout_rate = pre_defined_values[version]
        return depth_factor, width_factor, dropout_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        #create model from the base model
        for expand_ratio, channels, layers, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels*width_factor) / 4) #to make sure always divisible by 4
            layers_repeats = ceil(layers * depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride = stride if layer == 0 else 1,#stride only on the first layer of the small building block
                        kernel_size=kernel_size,
                        padding=kernel_size//2, # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )
                in_channels = out_channels

        features.append(
            CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )

        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))

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

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for version in pre_trained_weights_url:
        _, _, res, _ = pre_defined_values[version]
        num_examples, num_classes = 4, 10
        x = torch.randn((num_examples, 3, res, res)).to(device)
        model = EfficientNet(
            version=version,
            num_classes=num_classes,
        ).to(device)
        print(model(x).shape) # (num_examples, num_classes)

#API function for building our LPR model
def Get_EfficientNetB0(num_output, pretrainedPath):
    model = EfficientNet("b0", num_output)
    if pretrainedPath is not None:
        the_url = pre_trained_weights_url["b0"]
        b0_state_dict = model_zoo.load_url(the_url, model_dir=pretrainedPath)
        weights_load = {}
        state = model.state_dict()
        state_keys = list(state.keys())
        for i, the_keys in enumerate(b0_state_dict.keys()):
            print(the_keys,state_keys[i])
            if state_keys[i].find('features') == -1:
                continue #ignore the classifier part
            weights_load[state_keys[i]] = b0_state_dict[the_keys]
        state.update(weights_load)
        model.load_state_dict(state)
    return model

def Get_EfficientNetB1(num_output, pretrainedPath):
    model = EfficientNet("b1", num_output)
    if pretrainedPath is not None:
        the_url = pre_trained_weights_url["b1"]
        b0_state_dict = model_zoo.load_url(the_url, model_dir=pretrainedPath)
        weights_load = {}
        state = model.state_dict()
        state_keys = list(state.keys())
        print(len(state), len(b0_state_dict))
        for i, the_keys in enumerate(b0_state_dict.keys()):
            print(the_keys,state_keys[i])
            if state_keys[i].find('features') == -1:
                continue #ignore the classifier part
            weights_load[state_keys[i]] = b0_state_dict[the_keys]
        state.update(weights_load)
        model.load_state_dict(state)
    return model

def Get_EfficientNetB4(num_output, pretrainedPath):
    model = EfficientNet("b3", num_output)
    if pretrainedPath is not None:
        the_url = pre_trained_weights_url["b3"]
        b0_state_dict = model_zoo.load_url(the_url, model_dir=pretrainedPath)
        weights_load = {}
        state = model.state_dict()
        print(len(state), len(b0_state_dict))

        state_keys = list(state.keys())
        for i, the_keys in enumerate(b0_state_dict.keys()):
            print(the_keys,state_keys[i])
            if state_keys[i].find('features') == -1:
                continue #ignore the classifier part
            weights_load[state_keys[i]] = b0_state_dict[the_keys]
        state.update(weights_load)
        model.load_state_dict(state)
    return model

if __name__ == "__main__":
    # test()
    Get_EfficientNetB1(2, "weights/")