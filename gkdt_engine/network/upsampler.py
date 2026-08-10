import torch
import torch.nn as nn


'''
# An example copied from ViTPose
TopdownHeatmapSimpleHead(
  (loss): JointsMSELoss(
    (criterion): MSELoss()
  )
  (deconv_layers): Sequential(
    (0): ConvTranspose2d(1280, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
  (final_layer): Conv2d(256, 17, kernel_size=(1, 1), stride=(1, 1))
)
'''

class UpsamplerByDeconv(nn.Module):
    def __init__(self, 
                 in_channels,
                 num_deconv_layers=2,
                 num_deconv_filters=(256, 256),
                 num_deconv_kernels=(4, 4),
                 ):
        super().__init__()
        self.in_channels = in_channels
        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
        
        self.init_weights()


    def forward(self, x):
        """Forward function."""
        x = self.deconv_layers(x)
        return x

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding
    
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
