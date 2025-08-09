import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Linear, Softmax, Conv3d
from torch.nn.functional import upsample

class PAMEnc_3d(Module):

    def __init__(self, in_channels, norm_layer=nn.BatchNorm3d):
        super(PAMEnc_3d, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.pool2 = nn.AdaptiveAvgPool3d((1, 2, 2))
        self.pool3 = nn.AdaptiveAvgPool3d((1, 3, 3))
        self.pool4 = nn.AdaptiveAvgPool3d((1, 6, 6))

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), bias=False),
            norm_layer(in_channels),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), bias=False),
            norm_layer(in_channels),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), bias=False),
            norm_layer(in_channels),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), bias=False),
            norm_layer(in_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        b, c, t, h, w = x.size()

        feat1 = self.conv1(self.pool1(x)).view(b, c, -1)
        feat2 = self.conv2(self.pool2(x)).view(b, c, -1)
        feat3 = self.conv3(self.pool3(x)).view(b, c, -1)
        feat4 = self.conv4(self.pool4(x)).view(b, c, -1)

        return torch.cat((feat1, feat2, feat3, feat4), 2)


class PAMDec_3d(Module):

    def __init__(self, in_channels):
        super(PAMDec_3d, self).__init__()
        self.softmax = Softmax(dim=-1)
        self.scale = Parameter(torch.zeros(1))

        self.conv_query = Conv3d(in_channels=in_channels, out_channels=in_channels // 4,
                                 kernel_size=(1, 1, 1))  # query_conv3
        self.conv_key = Linear(in_channels, in_channels // 4)  # key_conv3
        self.conv_value = Linear(in_channels, in_channels)  # value3

    def forward(self, x, y):
        """
            inputs :
                x : input feature(N,C,T,H,W) y:gathering centers(N,K,M)
            returns :
                out : compact position attention feature
                attention map: (T*H*W)*M
        """
        m_batchsize, C, T, width, height = x.size()
        m_batchsize, K, M = y.size()

        proj_query = self.conv_query(x).view(m_batchsize, -1, T * width * height).permute(0, 2, 1)  # BxNxd
        proj_key = self.conv_key(y).view(m_batchsize, K, -1).permute(0, 2, 1)  # BxdxK
        energy = torch.bmm(proj_query, proj_key)  # BxNxK
        attention = self.softmax(energy)  # BxNxk

        proj_value = self.conv_value(y).permute(0, 2, 1)  # BxCxK
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # BxCxN
        out = out.view(m_batchsize, C, T, width, height)
        out = self.scale * out + x
        return out


class CAMDec_3d(nn.Module):
    def __init__(self):
        super(CAMDec_3d, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        """
        inputs:
            x: input feature(B, C, T, H, W)
            y: gathering centers(B, K, T, H, W)
        returns:
            out: compact channel attention feature
            attention map: K*C
        """
        m_batchsize, C, T, width, height = x.size()
        x_reshape = x.view(m_batchsize, C, -1)

        B, K, T, W, H = y.size()
        y_reshape = y.view(B, K, -1)
        proj_query = x_reshape  # BxCx(TxN)
        proj_key = y_reshape.permute(0, 2, 1)  # Bx(TxN)xC
        energy = torch.bmm(proj_query, proj_key)  # BxCxC
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = y.view(B, K, -1)  # BxKx(TxN)

        out = torch.bmm(attention, proj_value)  # BxCx(TxN)
        out = out.view(m_batchsize, C, T, width, height)

        out = x + self.scale * out
        return out


class TAMDec_3d(nn.Module):

    def __init__(self):
        super(TAMDec_3d, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        """
        inputs:
            x: input feature(B, C, T, H, W)
            y: gathering centers(B, K, T, H, W)
        returns:
            out: compact channel attention feature
            attention map: K*T
        """

        x = x.permute(0, 2, 1, 3, 4)
        m_batchsize, T, C, width, height = x.size()
        x_reshape = x.contiguous().view(m_batchsize, T, -1)

        y = y.permute(0, 2, 1, 3, 4)
        B, K, C, W, H = y.size()
        y_reshape = y.contiguous().view(B, K, -1)

        proj_query = x_reshape  # BxTx(CxN)
        proj_key = y_reshape.permute(0, 2, 1)  # Bx(CxN)xT
        energy = torch.bmm(proj_query, proj_key)  # BxTxT
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = y.contiguous().view(B, K, -1)  # BxKx(CxN)
        out = torch.bmm(attention, proj_value)  # BxTx(CxN)
        out = out.view(m_batchsize, T, C, width, height)
        out = x + self.scale * out

        out = out.permute(0, 2, 1, 3, 4)
        return out




class AND(Module):
    """
    Attentive Noise Decoding module
    """

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm3d):
        super(AND, self).__init__()

        inter_channels = 32
        self.conv_low = nn.Sequential(nn.Conv3d(in_channels, inter_channels, (3, 3, 3), padding=(1, 1, 1), bias=False),
                                      norm_layer(inter_channels),
                                      nn.ReLU())  # skipconv

        self.conv_cat = nn.Sequential(
            nn.Conv3d(in_channels + inter_channels, in_channels, (3, 3, 3), padding=(1, 1, 1), bias=False),
            norm_layer(in_channels),
            nn.ReLU())  # fusion1

        self.conv_att = nn.Sequential(nn.Conv3d(in_channels + inter_channels, 1, (1, 1, 1)),
                                      nn.Sigmoid())  # att

        self.conv_out = nn.Sequential(nn.Conv3d(in_channels, out_channels, (3, 3, 3), padding=(1, 1, 1), bias=False),
                                      norm_layer(out_channels),
                                      nn.ReLU())  # fusion2
        self.conv_final = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, (5, 1, 1), stride=(5, 1, 1), padding=0, bias=False),
            norm_layer(out_channels),
            nn.ReLU())
        self._up_kwargs = {'mode': 'trilinear', 'align_corners': True}

        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x, y):
        """
            inputs :
                x : low level feature(N,C,T,H,W)  y:high level feature(N,C,T,H,W)
            returns :
                out :  cross-level gating decoder feature (N,1,H,W)
        """
        low_lvl_feat = self.conv_low(x)
        high_lvl_feat = upsample(y, low_lvl_feat.size()[2:], **self._up_kwargs)
        feat_cat = torch.cat([low_lvl_feat, high_lvl_feat], 1)

        low_lvl_feat_refine = self.gamma * self.conv_att(feat_cat) * low_lvl_feat
        low_high_feat = torch.cat([low_lvl_feat_refine, high_lvl_feat], 1)
        low_high_feat = self.conv_cat(low_high_feat)
        low_high_feat = self.conv_out(low_high_feat)
        low_high_feat = self.conv_final(low_high_feat).squeeze(2)

        return low_high_feat


class CAF(Module):
    """ Cross-modality Attentive Fusion module"""

    def __init__(self, in_channels, norm_layer=nn.BatchNorm3d):
        super(CAF, self).__init__()
        inter_channels = in_channels // 4

        # PA
        self.pam_encoder = PAMEnc_3d(inter_channels, norm_layer)
        self.pam_decoder = PAMDec_3d(inter_channels)

        # CA
        self.cam_encoder = nn.Sequential(
            nn.Conv3d(inter_channels, inter_channels // 16, 1, bias=False),
            norm_layer(inter_channels // 16),
            nn.ReLU())
        self.cam_decoder = CAMDec_3d()

        # TA
        self.tam_encoder = nn.Sequential(
            nn.Conv3d(inter_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU())
        self.tam_decoder = TAMDec_3d()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            norm_layer(inter_channels),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv3d(inter_channels, inter_channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            norm_layer(inter_channels),
            nn.ReLU())

        self.conv_tam = nn.Sequential(
            nn.Conv3d(inter_channels, inter_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0), bias=False),
            norm_layer(inter_channels),
            nn.ReLU())

        self.conv_cat = nn.Sequential(
            nn.Conv3d(inter_channels * 3, in_channels, (3, 1, 1), padding=(1, 0, 0), bias=False),
            norm_layer(in_channels),
            nn.ReLU())

    def forward(self, x1, x2):
        """
            inputs :
                x1 : input feature maps [B, C, T, H, W]
                x2 : input feature maps [B, C, T, H, W]
            returns :
                out1 : PAM feature1 + CAM feature2 + TAM feature [B, C, T, H, W]
                out2 : PAM feature2 + CAM feature1 + TAM feature [B, C, T, H, W]
        """

        # x1 PAM
        x1_pa_1 = self.conv1(x1)
        x1_pa_2 = self.pam_encoder(x1_pa_1).permute(0, 2, 1)
        x1_pa_feat = self.conv2(self.pam_decoder(x1_pa_1, x1_pa_2))

        # x1 CAM
        x1_ca_1 = self.conv1(x1)
        x1_ca_2 = self.cam_encoder(x1_ca_1)
        x1_ca_feat = self.conv2(self.cam_decoder(x1_ca_1, x1_ca_2))

        # x2 PAM
        x2_pa_1 = self.conv1(x2)
        x2_pa_2 = self.pam_encoder(x2_pa_1).permute(0, 2, 1)
        x2_pa_feat = self.conv2(self.pam_decoder(x2_pa_1, x2_pa_2))

        # x2 CAM
        x2_ca_1 = self.conv1(x2)
        x2_ca_2 = self.cam_encoder(x2_ca_1)
        x2_ca_feat = self.conv2(self.cam_decoder(x2_ca_1, x2_ca_2))

        # TAM
        x_ta_1 = self.conv1(torch.cat([x1, x2], 2))
        x_ta_2 = self.tam_encoder(x_ta_1)
        x_ta_feat = self.conv_tam(self.tam_decoder(x_ta_1, x_ta_2))

        out1 = self.conv_cat(torch.cat([x1_pa_feat, x2_ca_feat, x_ta_feat], 1))
        out2 = self.conv_cat(torch.cat([x2_pa_feat, x1_ca_feat, x_ta_feat], 1))
        return out1, out2
