import torch
import torch.nn as nn


class ViewTransform(nn.Module):
    def __init__(self, bev_size=(256, 128), pixel_size=0.5):
        """
        View Transform module.

        Args:
            bev_size (tuple): Size of the BEV map (H, W).
            pixel_size (float): Real-world distance per pixel in BEV (m/pixel).
        """
        super().__init__()
        self.bev_size = bev_size
        self.pixel_size = pixel_size
        self.bev_projector = BEVProjector(bev_size, pixel_size)

    def forward(self, feature_maps, intrinsics, extrinsics):
        """
        Projects feature maps into BEV space.

        Args:
            feature_maps (torch.Tensor): Input feature maps (B, N, C, H, W).
            intrinsics (torch.Tensor): Intrinsic matrices (B, N, 3, 3).
            extrinsics (torch.Tensor): Extrinsic matrices (B, N, 4, 4).

        Returns:
            torch.Tensor: BEV map (B, C, H_bev, W_bev).
        """
        batch_size, num_cameras, channels, img_h, img_w = feature_maps.shape

        # Initialize BEV map
        bev_feature_map = torch.zeros(
            (batch_size, channels, *self.bev_size), device=feature_maps.device
        )

        for cam_idx in range(num_cameras):
            # Project each camera's feature map to BEV
            bev_feature_map = self.bev_projector(
                feature_maps[:, cam_idx],
                intrinsics[:, cam_idx],
                extrinsics[:, cam_idx],
                bev_feature_map,
            )

        return bev_feature_map


class BEVProjector(nn.Module):
    def __init__(self, bev_size=(256, 128), pixel_size=0.5):
        """
        BEV Projector module.

        Args:
            bev_size (tuple): Size of the BEV map (H, W).
            pixel_size (float): Real-world distance per pixel in BEV (m/pixel).
        """
        super().__init__()
        self.bev_size = bev_size
        self.pixel_size = pixel_size

    def forward(self, feature_map, intrinsic, extrinsic, bev_feature_map):
        """
        Generates the BEV map.

        Args:
            feature_map (torch.Tensor): Feature map from a single camera (B, C, H, W).
            intrinsic (torch.Tensor): Intrinsic matrix (B, 3, 3).
            extrinsic (torch.Tensor): Extrinsic matrix (B, 4, 4).
            bev_feature_map (torch.Tensor): Existing BEV feature map.

        Returns:
            torch.Tensor: Updated BEV feature map.
        """
        batch_size, channels, img_h, img_w = feature_map.shape

        # Step 1: Generate pixel coordinates
        pixel_coords = self.create_pixel_coords(img_h, img_w, feature_map.device)

        # Step 2: Transform pixel coordinates to camera coordinates
        cam_coords = self.pixel_to_camera_coords(pixel_coords, intrinsic)

        # Step 3: Transform camera coordinates to world coordinates
        world_coords = self.camera_to_world_coords(cam_coords, extrinsic)

        # Step 4: Transform world coordinates to BEV coordinates
        bev_coords = self.world_to_bev_coords(world_coords)

        # Step 5: Assign features to BEV map
        bev_feature_map = self.assign_features_to_bev(
            bev_coords, feature_map, bev_feature_map
        )

        return bev_feature_map

    def create_pixel_coords(self, img_h, img_w, device):
        """
        Creates pixel coordinates.

        Args:
            img_h, img_w (int): Image dimensions (H, W).
            device: Device to assign tensors.

        Returns:
            torch.Tensor: Pixel coordinates (3, H * W).
        """
        y, x = torch.meshgrid(
            torch.linspace(0, img_h - 1, img_h, device=device),
            torch.linspace(0, img_w - 1, img_w, device=device),
            indexing="ij",
        )
        ones = torch.ones_like(x)
        return torch.stack([x, y, ones], dim=0).reshape(3, -1)

    def pixel_to_camera_coords(self, pixel_coords, intrinsic):
        """
        Transforms pixel coordinates to camera coordinates.

        Args:
            pixel_coords (torch.Tensor): Pixel coordinates (3, H * W).
            intrinsic (torch.Tensor): Intrinsic matrix (B, 3, 3).

        Returns:
            torch.Tensor: Camera coordinates (3, H * W).
        """
        intrinsic_inv = torch.inverse(intrinsic)
        return intrinsic_inv @ pixel_coords

    def camera_to_world_coords(self, cam_coords, extrinsic):
        """
        Transforms camera coordinates to world coordinates.

        Args:
            cam_coords (torch.Tensor): Camera coordinates (3, H * W).
            extrinsic (torch.Tensor): Extrinsic matrix (B, 4, 4).

        Returns:
            torch.Tensor: World coordinates (3, H * W).
        """
        cam_coords_h = torch.cat([cam_coords, torch.ones_like(cam_coords[:1])], dim=0)
        world_coords_h = extrinsic @ cam_coords_h
        return world_coords_h[:3]

    def world_to_bev_coords(self, world_coords):
        """
        Transforms world coordinates to BEV coordinates.

        Args:
            world_coords (torch.Tensor): World coordinates (3, H * W).

        Returns:
            torch.Tensor: BEV coordinates (2, H * W).
        """
        x, y, z = world_coords
        bev_x = (x / self.pixel_size).long() + self.bev_size[1] // 2
        bev_y = (y / self.pixel_size).long() + self.bev_size[0] // 2
        return torch.stack([bev_x, bev_y], dim=0)

    def assign_features_to_bev(self, bev_coords, feature_map, bev_feature_map):
        """
        Assigns feature map values to BEV coordinates.

        Args:
            bev_coords (torch.Tensor): BEV coordinates (2, H * W).
            feature_map (torch.Tensor): Image feature map (B, C, H, W).
            bev_feature_map (torch.Tensor): Existing BEV feature map.

        Returns:
            torch.Tensor: Updated BEV feature map.
        """
        for i in range(bev_coords.shape[1]):
            bev_x, bev_y = bev_coords[:, i]
            if 0 <= bev_x < self.bev_size[1] and 0 <= bev_y < self.bev_size[0]:
                bev_feature_map[:, :, bev_y, bev_x] += feature_map[
                    :, :, i // feature_map.shape[3], i % feature_map.shape[3]
                ]
        return bev_feature_map
