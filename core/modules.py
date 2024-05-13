import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, inChannels=64, outChannels=64):
        super(ResBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.PReLU(),  # PReLU have learnable parameter, so didn't have inplace attribute
            nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outChannels)
        )

    def forward(self, x):
        identity = x
        output = self.resblock(x)
        output += identity
        return output


class upsampler(nn.Module):
    def __init__(self, up_scale,in_channels=64):
        super(upsampler, self).__init__()
        layers = []
        single_upsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        up_scale = up_scale//2
        for i in range(up_scale):
            layers.append(single_upsample)
        self.upsample_module = nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample_module(x)