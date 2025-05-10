import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    간단히 2번의 Conv + ReLU + Dropout 구성. 필요하면 BatchNorm 등을 추가할 수도 있음.
    """
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
    """
    UNet++의 'full' 버전: x0_4까지 모든 skip connection을 구현.
    - 5개의 down-sampling (레벨0 ~ 레벨4)
    - 논문 그림에 나오는 x2_2, x1_3, x0_4 등 포함
    """
    def __init__(self, 
                 in_channels=3,      # 입력 채널 수
                 out_channels=1,     # 최종 분류 채널 수 (Binary segmentation이면 1, Multi-class면 원하는 값)
                 filters=[64,128,256,512,1024],
                 dropout_p=0.3,
                 deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 예: filters = [64, 128, 256, 512, 1024]
        nb_filter = filters

        # (1) encoder 단계 conv들
        self.conv0_0 = ConvBlock(in_channels,     nb_filter[0], dropout_p)
        self.conv1_0 = ConvBlock(nb_filter[0],    nb_filter[1], dropout_p)
        self.conv2_0 = ConvBlock(nb_filter[1],    nb_filter[2], dropout_p)
        self.conv3_0 = ConvBlock(nb_filter[2],    nb_filter[3], dropout_p)
        self.conv4_0 = ConvBlock(nb_filter[3],    nb_filter[4], dropout_p)

        # (2) nested skip의 중간 단계 conv들
        # x0_1, x1_1, x2_1, x3_1
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0], dropout_p)
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1], dropout_p)
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2], dropout_p)
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3], dropout_p)

        # x0_2, x1_2, x2_2
        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], dropout_p)
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], dropout_p)
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2], dropout_p)

        # x0_3, x1_3
        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], dropout_p)
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1], dropout_p)

        # x0_4
        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0], dropout_p)

        # (3) Deep Supervision용 최종 1×1 Conv들
        if self.deep_supervision:
            # 분기별 결과를 뽑아낼 Conv, ex) output1 ~ output4
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            # single output
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        인덱스 표기는 논문 그림에 맞춰 x0_0, x1_0 ... x0_4 등으로 기록.
        """
        # -------------------------------------------
        # 1) Encoder (down-sampling) 파트
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # -------------------------------------------
        # 2) Decoder의 nested skip connections
        # Level-1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        # Level-2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # Level-3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # Level-4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        # -------------------------------------------
        # 3) Deep Supervision 여부에 따른 출력
        if self.deep_supervision:
            # 논문처럼 x0_1, x0_2, x0_3, x0_4 네 분기의 결과를 모두 반환 (개별 loss 계산 가능)
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [out1, out2, out3, out4]
        else:
            # 만약 Deep Supervision을 안 쓸 경우엔 가장 마지막 레벨 x0_4만으로 출력
            out = self.final(x0_4)
            return out