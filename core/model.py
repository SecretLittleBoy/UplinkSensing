import torch.nn as nn
from .modules import ResBlock, upsampler


class encoder(nn.Module):
    """
        按序可分的layer共8层 -> total_layer = 2+sum(n_block)，如下:
            [input_layer,ResBlock*2,downsample_layer,ResBlock*4]
    """

    def __init__(self, start, end, inchannels=1, mid_channel=4, n_block=[2, 4], up_scale=4):
        super().__init__()
        total_layer = 2+sum(n_block)
        module_list = []

        assert start > -1, f'start:{start} should be larger than -1'
        if end == -1:
            end = total_layer - 1
        else:
            assert end > start and end < total_layer, f'end:{end} should be larger than start:{start} or less than total_layer:{total_layer}'

        _channel = mid_channel

        input_layer = nn.Sequential(
            nn.Conv2d(inchannels, out_channels=_channel, kernel_size=3, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(_channel),
            nn.PReLU()
        )
        module_list.append(input_layer)
        del input_layer

        backbone_layers = []
        for i in range(n_block[0]):
            backbone_layers.append(ResBlock(inChannels=_channel, outChannels=_channel))

        _channel = up_scale*mid_channel
        downsample_layer = nn.Sequential(
            nn.Conv2d(mid_channel, out_channels=_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(_channel),
            nn.PReLU()
        )
        backbone_layers.append(downsample_layer)
        for i in range(n_block[1]):
            backbone_layers.append(ResBlock(inChannels=_channel, outChannels=_channel))

        module_list += backbone_layers
        del backbone_layers

        self.encode_net = nn.Sequential(*module_list[start:end+1])
        del module_list

    def forward(self, x):
        output = self.encode_net(x)
        return output


class decoder(nn.Module):
    def __init__(self, outchannels=2, mid_channel=64, up_scale=4):
        super().__init__()
        self.input_layer = nn.Sequential(
            ResBlock(inChannels=mid_channel*up_scale, outChannels=mid_channel*up_scale),
            nn.Conv2d(in_channels=mid_channel*up_scale, out_channels=mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.upsampler = upsampler(in_channels=mid_channel, up_scale=up_scale)
        self.output_conv = nn.Conv2d(in_channels=mid_channel, out_channels=outchannels, kernel_size=5, stride=1, padding=2, bias=False)

    def forward(self, x):
        output = self.input_layer(x)
        output = self.upsampler(output)
        output = self.output_conv(output)
        return output


class Net(nn.Module):
    def __init__(self, inchannels=1, outchannels=2, mid_channel=4, up_scale=4, n_block=[2, 4]):
        super().__init__()
        total_layer = 2+sum(n_block)
        self.encoder = encoder(start=0, end=total_layer-1, inchannels=inchannels, mid_channel=mid_channel, n_block=n_block, up_scale=up_scale)
        self.decoder = decoder(outchannels=outchannels, mid_channel=mid_channel, up_scale=up_scale)
        self.channel_model = nn.Identity()

    def forward(self, x):
        output = self.encoder(x)
        output = self.channel_model(output)
        output = self.decoder(output)
        return output
