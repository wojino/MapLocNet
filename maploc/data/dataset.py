from typing import List, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MapLocDataset(Dataset):
    def __init__(self, samples: List[Dict], image_size=(128, 352), augmentations=None):
        """
        Initialize the MapLocDataset.

        Args:
            samples (List[Dict]): List of samples from the NuScenesDataset.
            image_size (tuple): Target image size (height, width).
            augmentations (transforms.Compose, optional): Augmentations to apply to the images.
        """
        self.samples = samples
        self.image_size = image_size
        self.augmentations = augmentations or self.default_transforms()

    def default_transforms(self):
        """
        Define the default image transformations, including resizing and normalization.

        Returns:
            transforms.Compose: Default transformation pipeline.
        """
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def augmentation_transforms(self):
        """
        Define augmentation transformations, such as random cropping, flipping, and dropping.

        Returns:
            transforms.Compose: Augmentation transformation pipeline.
        """
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieve a sample and process it.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict: Processed image tensors and metadata.
        """
        sample = self.samples[idx]

        # Load images for all sensors
        sensors = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]

        images = {}
        for sensor in sensors:
            image_path = sample[sensor]["filename"]
            image = transforms.functional.pil_to_tensor(
                image_path
            )  # Placeholder for actual image loading
            if self.augmentations:
                image = self.augmentations(image)
            images[sensor] = image

        metadata = {
            "sample_token": sample["token"],
            "timestamp": sample["timestamp"],
        }

        return {"images": images, "metadata": metadata}


# Example usage with NuScenesDataModule
if __name__ == "__main__":
    from nuscenes.data_module import NuScenesDataModule
    from omegaconf import OmegaConf

    # Load config and data module
    cfg = OmegaConf.load("path/to/nuscenes.yaml")
    data_module = NuScenesDataModule(cfg)
    data_module.setup(stage="fit")

    # Initialize MapLocDataset with train samples
    train_dataset = MapLocDataset(data_module.train_dataset.samples)
    sample = train_dataset[0]  # Retrieve the first processed sample
    print(sample)
