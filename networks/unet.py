import torch
import torch.nn as nn


class double_convolution(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(double_convolution, self).__init__()
       
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class upsampling_block(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(upsampling_block, self).__init__()
        self.upsampling=nn.ConvTranspose2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, up):
        upsampled = self.upsampling(up)
        return upsampled
        # return self.up_conv(upsampled)


class UNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=1, filters=64, output_activation='sigmoid'):
        super(UNet, self).__init__()

        self.output_activation = output_activation.lower()

        self.pool = nn.MaxPool2d(2)

        self.inp = double_convolution(num_channels, filters)

        self.downsampling_1 = double_convolution(filters, filters*2)
        self.downsampling_2 = double_convolution(filters*2, filters*4)
        self.downsampling_3 = double_convolution(filters*4, filters*8)
        self.downsampling_4 = double_convolution(filters*8, filters*16)

        self.upsample4 = upsampling_block(filters*16, filters*8)
        self.upsampling_convolution4 = double_convolution(filters*16, filters*8)
        self.upsample3 = upsampling_block(filters*8, filters*4)
        self.upsampling_convolution3 = double_convolution(filters*8, filters*4)
        self.upsample2 = upsampling_block(filters*4, filters*2)
        self.upsampling_convolution2 = double_convolution(filters*4, filters*2)
        self.upsample1 = upsampling_block(filters*2, filters)
        self.upsampling_convolution1 = double_convolution(filters*2, filters)

        self.output = nn.Conv2d(filters, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        inp = self.inp(x)

        # downsampling path
        d1 = self.pool(inp)
        d1 = self.downsampling_1(d1)
        d2 = self.pool(d1)
        d2 = self.downsampling_2(d2)
        d3 = self.pool(d2)
        d3 = self.downsampling_3(d3)
        d4 = self.pool(d3)
        d4 = self.downsampling_4(d4)

        # upsampling path
        u4 = self.upsample4(d4)
        cat4 = torch.cat((d3, u4), dim=1)
        up4 = self.upsampling_convolution4(cat4)
        u3 = self.upsample3(up4)
        cat3 = torch.cat((d2, u3), dim=1)
        up3 = self.upsampling_convolution3(cat3)
        u2 = self.upsample2(up3)
        cat2 = torch.cat((d1, u2), dim=1)
        up2 = self.upsampling_convolution2(cat2)
        u1 = self.upsample1(up2)
        cat1 = torch.cat((inp, u1), dim=1)
        up1 = self.upsampling_convolution1(cat1)

        out = self.output(up1)

        if self.output_activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.output_activation == 'softmax':
            import torch.nn.functional as F
            return F.softmax(out, dim=1)
        else:
            raise NotImplementedError('Unknown output activation function')

def init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data,0.25,nonlinearity='relu')
        nn.init.constant(module.bias.data,0)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()

UNet2D = UNet()
# print(UNet2D)
total = sum([param.nelement() for param in UNet2D.parameters()])
print('total_param',total)
# for name,param in UNet2D.named_parameters():
#     print(name,param.size())
UNet2D.apply(init)
