import torch.nn as nn

import segmentation_models_pytorch as smp
import torch

# NOTE: Removed head_kernel as that's not a valid parameter from the codebase?
# Could be a custom model / implementation, not sure. Ask IG.
class UNet_Resnet18(nn.Module):
    def __init__(self, n_channels,n_classes,encoder_name='resnet18', encoder_weights=None):
        super(UNet_Resnet18, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.model = smp.Unet(encoder_name,
                              in_channels=n_channels,
                              classes=n_classes,
                              encoder_weights=encoder_weights,
                              encoder_depth=5,
                              decoder_channels=[256,128, 64, 32, 16])


    def forward(self,x):
        outputs = self.model(x)
        return outputs


if __name__ == '__main__':

    SMP = UNet_Resnet18(1,7)

    for name, param in SMP.named_parameters():
        print(name, param.shape)
