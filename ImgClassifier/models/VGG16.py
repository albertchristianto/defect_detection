import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class VGG16(nn.Module):
    def __init__(self, num_output, inp_size):
        """
        Model Definition
        """
        super(VGG16, self).__init__()
        feature_size = int(inp_size / 32) #because the features extractor has 5 maxpool layer
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * feature_size * feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_output)
        )

        self._initialize_weights_norm()

    def forward(self, inp):
        """
        Defines the forward computation performed at every call by defined VGG 16 network
        """
        out = self.features(inp)
        out = out.view(out.size(0), -1)  # linearized the output of the module 'features'
        out = self.classifier(out)
        return out

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

#API function for building our LPR model
def Get_VGG16(num_output, inp_size, pretrainedPath):
    model = VGG16(num_output, inp_size)
    if pretrainedPath is not None:
        vgg16_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        vgg_state_dict = model_zoo.load_url(vgg16_url, model_dir=pretrainedPath)
        weights_load = {}
        for the_keys in vgg_state_dict.keys():
            if the_keys.find('features') == -1:
                continue #ignore the classifier part
            weights_load[the_keys] = vgg_state_dict[the_keys]
        state = model.state_dict()
        state.update(weights_load)
        model.load_state_dict(state)
    return model