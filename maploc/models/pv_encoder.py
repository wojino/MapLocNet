import torch
import torch.nn as nn
import torchvision.models as models


class PVEncoder(nn.Module):
    def __init__(self, pretrained=True):
        """
        Initialize the PV Encoder.

        Args:
            pretrained (bool): Whether to use the pretrained EfficientNet-B0 model.
        """
        super(PVEncoder, self).__init__()

        # EfficientNet-B0 backbone network
        self.backbone = models.efficientnet_b0(pretrained=pretrained).features
        self.out_channels = 1280  # Number of output channels from EfficientNet-B0

        # 1x1 convolution to reduce the number of channels
        self.conv1x1 = nn.Conv2d(self.out_channels, 256, kernel_size=1)
        self.out_channels = 256  # Final output channels set to 256

    def forward(self, x):
        """
        Forward pass of the PV Encoder.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 3, H, W)

        Returns:
            torch.Tensor: Extracted feature map with shape (B, 256, H_out, W_out)
        """
        x = self.backbone(x)  # Extract features using EfficientNet-B0
        x = self.conv1x1(x)  # Reduce the number of channels using 1x1 convolution
        return x


# Example usage
if __name__ == "__main__":
    encoder = PVEncoder(pretrained=True)
    dummy_input = torch.randn(4, 3, 128, 352)  # Batch size of 4, image size 128x352
    features = encoder(dummy_input)
    print(f"Output shape: {features.shape}")
