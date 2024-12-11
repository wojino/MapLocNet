import torch
import torch.nn as nn
from ..models import PVEncoder, ViewTransform, BEVEncoder, BEVDecoder


class BEVModule(nn.Module):
    def __init__(self, input_size=(128, 352), bev_size=(256, 256), pixel_size=0.5):
        """
        BEV Module initialization.

        Args:
            input_size (tuple): Input image size (H, W).
            bev_size (tuple): BEV map size (H_bev, W_bev).
            pixel_size (float): Meters per pixel in the BEV space.
        """
        super(BEVModule, self).__init__()
        self.pv_encoder = PVEncoder(pretrained=True)
        self.view_transform = ViewTransform(bev_size=bev_size, pixel_size=pixel_size)
        self.bev_encoder = BEVEncoder(
            input_channels=256
        )  # Output channels of PV Encoder
        self.bev_decoder = BEVDecoder(input_channels=256)

    def forward(self, images, intrinsics, extrinsics):
        """
        Forward function for the BEV Module.

        Args:
            images (torch.Tensor): Input images (B, N, 3, H, W).
            intrinsics (torch.Tensor): Intrinsic matrices (B, N, 3, 3).
            extrinsics (torch.Tensor): Extrinsic matrices (B, N, 4, 4).

        Returns:
            torch.Tensor: Final BEV output (B, C, H_bev, W_bev).
        """
        batch_size, num_cameras, _, img_h, img_w = images.shape

        # Step 1: Extract image features using PV Encoder
        pv_features = torch.zeros(
            (batch_size, num_cameras, 256, img_h // 32, img_w // 32),
            device=images.device,
        )
        for cam_idx in range(num_cameras):
            pv_features[:, cam_idx] = self.pv_encoder(images[:, cam_idx])

        # Step 2: Transform features into BEV space
        bev_features = self.view_transform(pv_features, intrinsics, extrinsics)

        # Step 3: Encode BEV features
        encoded_bev = self.bev_encoder(bev_features)

        # Step 4: Decode BEV features to higher resolution
        decoded_bev = self.bev_decoder(encoded_bev)

        return decoded_bev


# Usage example
if __name__ == "__main__":
    dummy_images = torch.randn(2, 6, 3, 128, 352)  # Batch=2, Cameras=6, Image=128x352
    dummy_intrinsics = torch.eye(3).repeat(2, 6, 1, 1)  # (B, N, 3, 3)
    dummy_extrinsics = torch.eye(4).repeat(2, 6, 1, 1)  # (B, N, 4, 4)

    bev_module = BEVModule()
    output = bev_module(dummy_images, dummy_intrinsics, dummy_extrinsics)
    print(f"Output Shape: {output.shape}")  # Expected: (B, C, H_bev, W_bev)
