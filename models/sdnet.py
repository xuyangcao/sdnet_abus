import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time

from layers.blocks import *
from layers.adain import *
from models.d2unet import D2UNet

class AdaINDecoder(nn.Module):
    def __init__(self, anatomy_out_channels):
        super().__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.conv1 = conv_relu(self.anatomy_out_channels, 128, 3, 1, 1)
        self.conv2 = conv_relu(128, 64, 3, 1, 1)
        self.conv3 = conv_relu(64, 32, 3, 1, 1)
        self.conv4 = conv_no_activ(32, 1, 3, 1, 1)

    def forward(self, a, z):
        out = adaptive_instance_normalization(a, z)
        out = self.conv1(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv2(out)
        out = adaptive_instance_normalization(out, z)
        out = self.conv3(out)
        out = adaptive_instance_normalization(out, z)
        out = F.tanh(self.conv4(out))

        return out


class Decoder(nn.Module):
    def __init__(self, anatomy_out_channels, z_length, num_mask_channels):
        super(Decoder, self).__init__()
        self.anatomy_out_channels = anatomy_out_channels
        self.z_length = z_length
        self.num_mask_channels = num_mask_channels
        self.decoder = AdaINDecoder(self.anatomy_out_channels)

    def forward(self, a, z):
        out = self.decoder(a, z)

        return out


class Segmentor(nn.Module):
    def __init__(self, num_output_channels, num_classes):
        super(Segmentor, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_classes = num_classes
        
        self.conv1 = conv_bn_relu(self.num_output_channels, 64, 3, 1, 1)
        self.conv2 = conv_bn_relu(64, 64, 3, 1, 1)
        self.pred = nn.Conv2d(64, self.num_classes, 1, 1, 0)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pred(out)
        out = F.softmax(out, dim=1)

        return out


class AEncoder(nn.Module):
    def __init__(self, width, height, ndf, num_output_channels, norm, upsample):
        super(AEncoder, self).__init__()
        """
        UNet encoder for the anatomy factors of the image
        num_output_channels: number of spatial (anatomy) factors to encode
        """
        self.width = width 
        self.height = height
        self.ndf = ndf
        self.num_output_channels = num_output_channels
        self.norm = norm
        self.upsample = upsample

        #self.unet = UNet(self.width, self.height, self.ndf, self.num_output_channels, self.norm, self.upsample)
        self.unet = D2UNet(in_channels=1, num_classes=self.num_output_channels, drop_rate=0.1, skip_connetcion=True)

    def forward(self, x):
        out = self.unet(x)
        out = F.gumbel_softmax(out, hard=True, dim=1)
        return out 


class MEncoder(nn.Module):
    def __init__(self, z_length):
        super(MEncoder, self).__init__()
        """
        VAE encoder to extract intensity (modality) information from the image
        z_length: length of the output vector
        """
        self.z_length = z_length

        self.block1 = conv_bn_lrelu(5, 16, 3, 2, 1)
        self.block2 = conv_bn_lrelu(16, 32, 3, 2, 1)
        self.block3 = conv_bn_lrelu(32, 64, 3, 2, 1)
        self.block4 = conv_bn_lrelu(64, 128, 3, 2, 1)
        self.fc = nn.Linear(32768, 32)
        #self.norm = nn.BatchNorm1d(32)
        self.activ = nn.LeakyReLU(0.03, inplace=True)
        self.mu = nn.Linear(32, self.z_length)
        self.logvar = nn.Linear(32, self.z_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def encode(self, x):
        return self.mu(x), self.logvar(x)

    def forward(self, a, x):
        out = torch.cat([a, x], 1)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.fc(out.view(-1, out.shape[1] * out.shape[2] * out.shape[3]))
        #out = self.norm(out)
        out = self.activ(out)

        mu, logvar = self.encode(out)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class SDNet(nn.Module):
    def __init__(self, width, height, num_classes, ndf, z_length, norm, upsample, anatomy_out_channels, num_mask_channels, **kwargs):
        super(SDNet, self).__init__()
        """
        Args:
            width: input width
            height: input height
            upsample: upsampling type (nearest | bilateral)
            num_classes: number of semantice segmentation classes
            z_length: number of modality factors
            anatomy_out_channels: number of anatomy factors
            norm: feature normalization method (BatchNorm)
            ndf: number of feature channels
        """
        self.h = height
        self.w = width
        self.ndf = ndf
        self.z_length = z_length
        self.anatomy_out_channels = anatomy_out_channels
        self.norm = norm
        self.upsample = upsample
        self.num_classes = num_classes
        self.num_mask_channels = num_mask_channels

        self.m_encoder = MEncoder(self.z_length)
        self.a_encoder = AEncoder(self.h, self.w, self.ndf, self.anatomy_out_channels, self.norm, self.upsample)
        self.segmentor = Segmentor(self.anatomy_out_channels, self.num_classes)
        self.decoder = Decoder(self.anatomy_out_channels, self.z_length, self.num_mask_channels)

    def forward(self, x, mask, script_type):
        a_out = self.a_encoder(x)
        seg_pred = self.segmentor(a_out)
        z_out, mu_out, logvar_out = self.m_encoder(a_out, x)
        if script_type == 'training':
            reco = self.decoder(a_out, z_out)
            _, mu_out_tilde, _ = self.m_encoder(a_out, x)
        else:
            reco = self.decoder(a_out, mu_out)
            mu_out_tilde = mu_out #dummy assignment, not needed during validation

        return reco, z_out, mu_out_tilde, mu_out_tilde, a_out, seg_pred, mu_out, logvar_out, mu_out, logvar_out
