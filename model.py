import torch
from uniformer import Encoder
from segformer_head import Decoder_3d
from base_model import BaseModel
from AttentionModule import AND


class TruVIL(BaseModel):
    def __init__(self):
        super(TruVIL, self).__init__()

        self.in_channels = [64, 128, 320, 512, 512]
        self.channels = 768
        self.in_index = [0, 1, 2, 3, 4]

        self.encoder = Encoder()
        self.decoder = Decoder_3d(channels=self.channels, in_channels=self.in_channels, in_index=self.in_index)
        self.AND = AND(in_channels=3, out_channels=1)

    def forward(self, inputs):
        x, x_res1 = self.encoder(inputs)
        x = self.decoder(x)
        out = self.AND(x_res1, x)
        return out

