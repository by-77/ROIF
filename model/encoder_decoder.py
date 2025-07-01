import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
from options import HiDDenConfiguration
from Noise_Layer import ScreenShooting
import torch
from PIL import Image
import os
class EncoderDecoder(nn.Module):

    def __init__(self, config: HiDDenConfiguration):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config)
        self.noiser = ScreenShooting()
        self.decoder = Decoder(config)
        self.average=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.linear=nn.Linear(config.message_length,config.message_length)

    def forward(self,batch):
        image, message=batch
        encoded_image= self.encoder(image, message)

        noised_and_cover = self.noiser(encoded_image)


        if isinstance(noised_and_cover, torch.Tensor):
            noised_and_cover = noised_and_cover.to(torch.float32)

        elif isinstance(noised_and_cover, (list, tuple)):
            noised_and_cover = [tensor.to(torch.float32) for tensor in noised_and_cover]

        noised_image= noised_and_cover
        decoded_message_m=self.decoder(noised_image)


        return encoded_image, noised_image,decoded_message_m
