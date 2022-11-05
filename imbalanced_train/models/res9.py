import torch.nn as nn
from torch import cat
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class resnet9(nn.Module):
    def __init__(self, num_classes, use_norm=None, add_embed=False, in_channels=3):
        super().__init__()

        self.add_embed = add_embed
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.flatten = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())

        if self.add_embed:
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(512 + 1, num_classes))
        else:
            self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, num_classes))

    def forward(self, out, is_real):
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.flatten(out)
        if self.add_embed:
            out = cat((out, is_real.view(-1, 1)), dim=1)
        out = self.classifier(out)
        return out