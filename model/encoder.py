import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.Dense_block import Bottleneck


class ASSA(nn.Module):


    def __init__(self, in_channels, reduction_ratio=16):
        super(ASSA, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)

        # Spatial attention
        sa = self.spatial_attention(x)

        # Combine both attentions
        out = x * ca * sa
        return out


class Encoder(nn.Module):

    def conv1(self, in_channel, out_channel):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         stride=1,
                         kernel_size=7, padding=3)

    def conv2(self, in_channel, out_chanenl):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_chanenl,
                         stride=1,
                         kernel_size=3,
                         padding=1)

    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        # Initialize ASSA modules
        self.assa1 = ASSA(self.conv_channels + config.message_length)
        self.assa2 = ASSA(self.conv_channels * 2 + config.message_length)
        self.assa3 = ASSA(self.conv_channels * 3 + config.message_length)
        self.assa_a1 = ASSA(self.conv_channels)
        self.assa_a2 = ASSA(self.conv_channels * 2)
        self.assa_a3 = ASSA(self.conv_channels * 3)

        self.first_layer = nn.Sequential(
            self.conv2(3, self.conv_channels)
        )

        self.second_layer = nn.Sequential(
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.third_layer = nn.Sequential(
            self.conv2(self.conv_channels * 2, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.fourth_layer = nn.Sequential(
            self.conv2(self.conv_channels * 3 + config.message_length, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.Dense_block1 = Bottleneck(self.conv_channels + config.message_length, self.conv_channels)
        self.Dense_block2 = Bottleneck(self.conv_channels * 2 + config.message_length, self.conv_channels)
        self.Dense_block3 = Bottleneck(self.conv_channels * 3 + config.message_length, self.conv_channels)
        self.Dense_block_a1 = Bottleneck(self.conv_channels, self.conv_channels)
        self.Dense_block_a2 = Bottleneck(self.conv_channels * 2, self.conv_channels)
        self.Dense_block_a3 = Bottleneck(self.conv_channels * 3, self.conv_channels)

        self.fivth_layer = nn.Sequential(
            nn.BatchNorm2d(self.conv_channels + config.message_length),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels + config.message_length, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, config.message_length),
        )
        self.sixth_layer = nn.Sequential(
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, self.conv_channels),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
            self.conv2(self.conv_channels, config.message_length),
            nn.Softmax(dim=1)
        )
        self.softmax = nn.Sequential(nn.Softmax(dim=1))

        self.final_layer = nn.Sequential(nn.Conv2d(config.message_length, 3, kernel_size=3, padding=1),
                                         )

    def forward(self, image, message):
        H, W = image.size()[2], image.size()[3]

        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1, -1, H, W)

        feature0 = self.first_layer(image)

        # Apply ASSA to features before dense blocks
        feature1_input = self.assa1(torch.cat((feature0, expanded_message), 1))
        feature1 = self.Dense_block1(feature1_input, last=True)

        feature2_input = self.assa2(torch.cat((feature0, expanded_message, feature1), 1))
        feature2 = self.Dense_block2(feature2_input, last=True)

        feature3_input = self.assa3(torch.cat((feature0, expanded_message, feature1, feature2), 1))
        feature3 = self.Dense_block3(feature3_input, last=True)

        feature3 = self.fivth_layer(torch.cat((feature3, expanded_message), 1))

        # Apply ASSA to attention path features
        feature_a1 = self.Dense_block_a1(self.assa_a1(feature0))
        feature_a2 = self.Dense_block_a2(self.assa_a2(torch.cat((feature0, feature_a1), 1)))
        feature_attention = self.Dense_block_a3(self.assa_a3(torch.cat((feature0, feature_a1, feature_a2), 1)),
                                                last=True)

        feature_mask = (self.sixth_layer(feature_attention)) * 30
        feature = feature3 * feature_mask
        im_w = self.final_layer(feature)
        im_w = im_w + image
        return im_w







