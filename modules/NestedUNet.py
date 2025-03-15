import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
    def forward(self, x):
        return self.conv(x)

class NestedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, deep_supervision=False, dropout_p=0.3, filters=[64, 128, 256, 512]):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        nb_filter = filters

        self.conv0_0 = ConvBlock(in_channels, nb_filter[0], dropout_p)
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1], dropout_p)
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2], dropout_p)
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3], dropout_p)
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[3]*2, dropout_p)

        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0], dropout_p)
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1], dropout_p)
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2], dropout_p)
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[3]*2, nb_filter[3], dropout_p)

        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], dropout_p)
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], dropout_p)

        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], dropout_p)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Decoder / Nested connections
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)]), dim=1)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)]), dim=1)

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)]), dim=1)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output = (output1 + output2 + output3) / 3.0
            return output
        else:
            output = self.final(x0_3)
            return output

