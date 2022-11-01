import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

backbone_features = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M", ],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}

vgg_url = {
    "VGG11":"https://download.pytorch.org/models/vgg11-8a719046.pth",
    "VGG13":"https://download.pytorch.org/models/vgg13-19584684.pth",
    "VGG16":"https://download.pytorch.org/models/vgg16-397923af.pth",
    "VGG19":"https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
}

class VGG(nn.Module):
    def __init__(self, inp_size=224, in_channels=3, num_classes=1000, model_type="VGG16"):
        super(VGG, self).__init__()
        feature_size = int(inp_size / 32) #because the features extractor has 5 maxpool layer
        self.in_channels = in_channels
        self.features = self.create_conv_layers(backbone_features[model_type])

        self.classifier = nn.Sequential(
            nn.Linear(512 * feature_size * feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        self._initialize_weights_norm()

    def forward(self, inp):
        out = self.features(inp)
        out = out.view(out.size(0), -1)  # linearized the output of the module 'features'
        out = self.classifier(out)
        return out

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

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

def Get_VGG(model_type, in_channels, num_output, inp_size, pretrainedPath):
    model = VGG(inp_size=inp_size, in_channels=in_channels, num_classes=num_output, model_type=model_type)
    if in_channels!=3:
            print("imput channels are not equal to 3 is not supported with loading pretrained weights")
            return model
    if pretrainedPath is not None and in_channels==3:
        the_url = vgg_url[model_type]
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Get_VGG("VGG11", 3, 2, 224, '.').to(device)