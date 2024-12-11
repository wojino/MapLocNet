class BEVDecoder(nn.Module):
    def __init__(self, output_channels):
        """
        Initialize the BEV Decoder with multi-scale fusion for coarse-to-fine strategy.

        Args:
            output_channels (int): Number of output feature channels.
        """
        super(BEVDecoder, self).__init__()
        self.upsample1 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.upsample2 = nn.ConvTranspose2d(
            256 + 256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.upsample3 = nn.ConvTranspose2d(
            128 + 128,
            output_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.relu = nn.ReLU()

    def forward(self, features):
        """
        Forward pass for BEV Decoder.

        Args:
            features (List[torch.Tensor]): Multi-scale feature maps from BEV Encoder.

        Returns:
            torch.Tensor: Decoded BEV feature map.
        """
        coarse = features[-1]  # Lowest resolution feature map
        x = self.relu(self.upsample1(coarse))

        # Fuse with intermediate feature map
        x = torch.cat([x, features[-2]], dim=1)
        x = self.relu(self.upsample2(x))

        # Fuse with highest resolution feature map
        x = torch.cat([x, features[-3]], dim=1)
        x = self.upsample3(x)

        return x
