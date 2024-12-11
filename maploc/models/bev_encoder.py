class BEVEncoder(nn.Module):
    def __init__(self, input_channels):
        """
        Initialize the BEV Encoder with multi-scale outputs for coarse-to-fine strategy.

        Args:
            input_channels (int): Number of input feature channels.
        """
        super(BEVEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    input_channels, 128, kernel_size=3, stride=2, padding=1
                ),  # Downsample 1
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample 2
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Downsample 3
                nn.ReLU(),
            ]
        )

    def forward(self, x):
        """
        Forward pass for BEV Encoder.

        Args:
            x (torch.Tensor): Input tensor (B, C, H, W).

        Returns:
            List[torch.Tensor]: Multi-scale feature maps.
        """
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)  # Save each scale's feature map
        return features
