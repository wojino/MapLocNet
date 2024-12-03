from pathlib import Path
from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from ... import logger, CONFIG_PATH


class NuScenesDataset(Dataset):
    def __init__(self, nusc: NuScenes, split: str, data_dir: Path):
        self.nusc = nusc
        self.split = split
        self.data_dir = data_dir
        self.samples = self._load_samples(split)

    def _load_samples(self, split: str) -> List:
        scene_splits = create_splits_scenes()
        samples = []
        for scene in self.nusc.scene:
            if scene["name"] in scene_splits[split]:
                token = scene["first_sample_token"]
                while token:
                    sample = self.nusc.get("sample", token)
                    samples.append(sample)
                    token = sample["next"]
        logger.info(f"Loaded {len(samples)} samples for {split} split.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sensors = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        sample = self.samples[idx]
        data = {}
        for sensor in sensors:
            data[sensor] = self.nusc.get("sample_data", sample["data"][sensor])
        return data


class NuScenesDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.version = self.cfg.version
        self.data_dir = self.cfg.data_dir
        self.batch_size = self.cfg.batch_size
        self.num_workers = self.cfg.num_workers
        self.nusc = NuScenes(version=self.version, dataroot=self.data_dir, verbose=True)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = NuScenesDataset(self.nusc, "train", self.data_dir)
            self.val_dataset = NuScenesDataset(self.nusc, "val", self.data_dir)
        elif stage == "test":
            self.test_dataset = NuScenesDataset(self.nusc, "test", self.data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    cfg = OmegaConf.load(CONFIG_PATH / "data/nuscenes.yaml")
    logger.debug(OmegaConf.to_yaml(cfg))
    data_module = NuScenesDataModule(cfg)
    data_module.setup(stage="fit")
