import torch
import torch.nn as nn
from models.enc.enc_9xx_pvt2 import pvt_v2_b2 as Pvt2Encoder
from models.dec.dec_9xx_unet__convmodule import UNet as UnetDecoder
from models.tem.tem_905_timeS import stt_b1 as STTransformer


class AnomalyDetection(nn.Module):
    def __init__(self, config, logger, is_trained=True):
        super(AnomalyDetection, self).__init__()
        pretrained = config.MODEL.PRETRAINED if is_trained else None

        # Encoder (PVTv2 backbone)
        self.encoder = Pvt2Encoder(pretrained=pretrained, logger=logger)

        # Decoder (U-Net 구조, 4-frame 입력)
        self.decoder = UnetDecoder(num_input_frames=4, embed_dim=[64, 128, 320, 512])

        # SpatioTemporal Transformer
        self.sst = STTransformer()

    def forward(self, x):
        features = []
        for xi in x:
            features.append(self.encoder(xi))

        # list of lists transpose
        features = list(map(list, zip(*features)))

        temporal = []
        for i in range(len(features)):
            temporal.append(torch.stack([fea for fea in features[i]], dim=1))

        stt = self.sst(temporal)

        for i in range(len(features)):
            features[i] = torch.cat([fea for fea in features[i]], dim=1)

        # features + temporal
        for i in range(len(features)):
            features[i] = features[i] + stt[i]

        # Decoder
        features = self.decoder(features)
        return features
