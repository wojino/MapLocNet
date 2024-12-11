import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from data.nuscenes.data_module import NuScenesDataModule
from models.bev_module import BEVModule


class BEVTrainer(pl.LightningModule):
    def __init__(self, config):
        """
        Trainer for the BEV Module.

        Args:
            config (OmegaConf): Configuration dictionary containing model and training parameters.
        """
        super().__init__()
        self.model = BEVModule(config)  # Initialize the BEV Module.
        self.criterion = (
            CrossEntropyLoss()
        )  # Define the loss function for semantic segmentation.
        self.learning_rate = (
            config.learning_rate
        )  # Set the learning rate from the config.

    def forward(self, images, intrinsics, extrinsics):
        """
        Forward pass through the BEV module.

        Args:
            images (torch.Tensor): Input surround-view images (B, N, C, H, W).
            intrinsics (torch.Tensor): Intrinsic camera parameters (B, N, 3, 3).
            extrinsics (torch.Tensor): Extrinsic camera parameters (B, N, 4, 4).

        Returns:
            torch.Tensor: BEV feature map.
        """
        return self.model(images, intrinsics, extrinsics)

    def training_step(self, batch, batch_idx):
        """
        Training step performed for each batch.

        Args:
            batch (tuple): A batch containing images, labels, intrinsics, and extrinsics.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss for the batch.
        """
        images, labels, intrinsics, extrinsics = batch  # Unpack the batch data.
        outputs = self(
            images, intrinsics, extrinsics
        )  # Forward pass through the model.
        loss = self.criterion(outputs, labels)  # Compute the loss.
        self.log("train_loss", loss)  # Log the training loss.
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate schedulers.

        Returns:
            List: Optimizer and scheduler.
        """
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-7
        )  # Define the optimizer.
        scheduler = CosineAnnealingLR(
            optimizer, T_max=200
        )  # Define the learning rate scheduler.
        return [optimizer], [scheduler]


# Load the configuration file
from omegaconf import OmegaConf

config = OmegaConf.load("configs/bev_config.yaml")

# Prepare the data module
data_module = NuScenesDataModule(config)

# Train the model
trainer = pl.Trainer(
    max_epochs=config.epochs, gpus=1
)  # Define the PyTorch Lightning Trainer.
trainer.fit(BEVTrainer(config), data_module)  # Start training.
