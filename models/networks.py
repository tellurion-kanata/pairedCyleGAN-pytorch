from .units import *

class DeepResidualNetwork(nn.Module):
    def __init__(self, src_channels, out_channels, F=False):
        super(DeepResidualNetwork, self).__init__()
        self.F = F

        if not self.F:
            self.src_conv = nn.Conv2d(src_channels, 64, kernel_size=7, padding=3)
            self.ref_conv = nn.Conv2d(out_channels, 64, kernel_size=7, padding=3)
        else:
            self.src_conv = nn.Conv2d(src_channels, 128, kernel_size=7, padding=3)

        self.res1 = nn.Sequential(*[
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        ])

        self.res2 = nn.Sequential(*[
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
        ])

        self.res3 = nn.Sequential(*[
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4)
        ])

        self.res4 = nn.Sequential(*[
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.Tanh()
        ])

    def forward(self, x, ref=None):
        y = self.src_conv(x)
        if not self.F:
           ref = self.ref_conv(ref)
           y = torch.cat((y, ref), 1)
        y = y + self.res1(y)
        y = y + self.res2(y)
        y = y + self.res3(y)
        y = self.res4(y)

        if not self.F:
            y = y + x
        return y


""" Default stage Discriminator
Downsampling factor: 2^n_layer + 1 (downsample = True)   defalut is 3
                     2^n_layer (downsample = False)
"""
class CustomDiscriminator(nn.Module):
    def __init__(self, in_channels, n_layers=3, ndf=64, downsample=False, spec_norm=False, use_bias=True):
        super(CustomDiscriminator, self).__init__()
        model = []
        model += [
            DisConvLayer(in_channels, 64, kernel_size=3, stride=1, padding=1, spec_norm=spec_norm, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            DisConvLayer(64, 64, kernel_size=3, stride=2, padding=1, spec_norm=spec_norm, bias=use_bias),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        prev = 64
        for n in range(1, n_layers):
            nf_mult = min(2 ** n, 8)
            model += [
                DisConvLayer(prev, ndf * nf_mult, kernel_size=3, stride=1, padding=1, spec_norm=spec_norm, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                DisConvLayer(ndf * nf_mult, ndf * nf_mult, kernel_size=3, stride=2, padding=1, spec_norm=spec_norm, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            ]
            prev = nf_mult * ndf

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        stride = 2 if downsample else 1
        model += [
            DisConvLayer(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=stride, padding=1, spec_norm=spec_norm, bias=use_bias),
            nn.LeakyReLU(0.2, True)
        ]

        model += [DisConvLayer(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1, spec_norm=spec_norm, bias=use_bias)]
        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)