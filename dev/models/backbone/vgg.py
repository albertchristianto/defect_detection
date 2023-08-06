import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

vgg_param = {
    "11":[[64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], "https://download.pytorch.org/models/vgg11-8a719046.pth"],
    "13":[[64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], "https://download.pytorch.org/models/vgg13-19584684.pth"],
    "16":[[64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M", ], "https://download.pytorch.org/models/vgg16-397923af.pth"],
    "19":[[64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",], "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"]
}

class VGG(nn.Module):
    def __init__(self, model_type="16"):
        super(VGG, self).__init__()
        self.in_channels = 3
        self.features = self.create_conv_layers(vgg_param[model_type][0])
        self._initialize_weights_norm()

    def forward(self, x):
        features = []
        for i, each_block in enumerate(self.features):
            x = each_block(x)
            if i in self.keep_features:
                features.append(x)
        features.append(x)
        return features


    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        self.keep_features = []
        self.out_features_size = [256, 512, 512]
        stride_now_tmp = 1
        keep_features_tmp = [8, 16]
        for x in architecture:
            if type(x) == int:
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=x,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
                stride_now_tmp *= 2
                if stride_now_tmp in keep_features_tmp:
                    self.keep_features.append(len(layers) - 1)

        return nn.Sequential(*layers)

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

def Get_VGG(model_type, pretrainedPath):
    model = VGG(model_type)
    if pretrainedPath is not None:
        the_url = vgg_param[model_type][1]
        vgg_state_dict = model_zoo.load_url(the_url, model_dir=pretrainedPath)
        weights_load = {}
        for the_keys in vgg_state_dict.keys():
            if the_keys.find('features') == -1:
                continue #ignore the classifier part
            weights_load[the_keys] = vgg_state_dict[the_keys]
        state = model.state_dict()
        state.update(weights_load)
        model.load_state_dict(state)
    return model

if __name__ == "__main__":
    model_type = '19'
    pretrainedPath = "weights/"
    model = Get_VGG(model_type, pretrainedPath)
    fake_input = torch.rand(1, 3, 224, 224, requires_grad=True)
    fake_output = model(fake_input)
    print(model_type, model.out_features_size, fake_output[0].size(), fake_output[1].size(), fake_output[2].size())