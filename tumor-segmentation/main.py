import os
import torch
import segmentation_models_pytorch as smp
import random
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import albumentations as A
import cv2
import wandb
from pytorch_lightning.callbacks import Callback, EarlyStopping

DESIRED_WIDTH = 416
DESIRED_HEIGHT = 992


class DiceLossThresholdEarlyStopping(Callback):
    """Custom callback to stop training if validation dice loss is not below threshold after specified epochs"""

    def __init__(self, threshold=0.8, check_epoch=25, monitor="valid_dice_loss"):
        self.threshold = threshold
        self.check_epoch = check_epoch
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module):
        # Check if we've reached the specified epoch
        if (
            trainer.current_epoch + 1 == self.check_epoch
        ):  # +1 because epochs are 0-indexed
            # Get the current validation dice loss
            current_metrics = trainer.logged_metrics
            if self.monitor in current_metrics:
                current_dice_loss = current_metrics[self.monitor].item()
                print(
                    f"\nEpoch {self.check_epoch}: Validation dice loss = {current_dice_loss:.4f}"
                )

                if current_dice_loss >= self.threshold:
                    print(
                        f"Stopping training: Validation dice loss ({current_dice_loss:.4f}) is not below {self.threshold} after {self.check_epoch} epochs"
                    )
                    trainer.should_stop = True
                else:
                    print(
                        f"Continuing training: Validation dice loss ({current_dice_loss:.4f}) is below {self.threshold}"
                    )
            else:
                print(f"Warning: Metric '{self.monitor}' not found in logged metrics")


def seed_worker(worker_id):
    """Initialize random seeds for each worker to ensure reproducibility while maintaining parallelism"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)


def read_image_as_numpy(path):
    """Load image as numpy array with shape (H, W)"""
    image = Image.open(path).convert("L")  # Convert to grayscale
    return np.array(image, dtype=np.uint8)


def get_all_in_folder(dir):
    output = []
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        output.append(read_image_as_numpy(path))
    return output


def get_images(image_dir, mask_dir, control_dir):
    images = []
    masks = []
    control = get_all_in_folder(control_dir)

    image_filenames = os.listdir(image_dir)
    mask_filenames = [
        filename.replace("patient", "segmentation") for filename in image_filenames
    ]
    for img_file, mask_file in zip(image_filenames, mask_filenames):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        images.append(read_image_as_numpy(img_path))
        masks.append(read_image_as_numpy(mask_path))

    return images, masks, control


def process_images(images, transform):
    """Apply transform to list of images, keep as list of numpy arrays"""
    return [transform(image) for image in images]


def process_data(images, masks, control, transform):
    """Process images, masks, and control - keep as numpy arrays"""
    images = process_images(images, transform)
    masks = process_images(masks, transform)
    control = process_images(control, transform)
    return images, masks, control


def image_transform(image):
    """Transform numpy image - just return original for now, padding will happen after augmentations"""
    return image


def pad_image_to_size(image, target_height, target_width):
    """Pad image to target size"""
    height, width = image.shape

    # Calculate padding needed
    bottom_pad = max(0, target_height - height)
    left_pad = max(0, int(np.floor((target_width - width) / 2)))
    right_pad = max(0, int(np.ceil((target_width - width) / 2)))

    # Apply padding
    image = np.pad(
        image,
        ((0, bottom_pad), (left_pad, right_pad)),
        mode="constant",
        constant_values=255,
    )

    # Crop if image is larger than desired size
    image = image[:target_height, :target_width]

    return image


def reverse_image_transform(image, width_original, height_original):
    image = image * 255
    bottom_pad = DESIRED_HEIGHT - height_original
    left_pad = np.floor((DESIRED_WIDTH - width_original) / 2).astype(int)
    right_pad = np.ceil((DESIRED_WIDTH - width_original) / 2).astype(int)
    image = image[
        :,
        :-bottom_pad,
        left_pad:-right_pad,
    ]
    image = image.repeat(3, 1, 1)
    return image


class CustomDataset(Dataset):
    def __init__(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        control: list[np.ndarray] | None = None,
        augmentation: A.Compose | None = None,
        control_prob: float = 0,
    ):
        self.images = images
        self.masks = masks
        self.control = control

        self.augmentation = augmentation
        self.control_prob = control_prob

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        should_control = (
            self.control is not None and random.random() < self.control_prob
        )

        if should_control:
            control_idx = random.randint(0, len(self.control) - 1)
            image = self.control[control_idx]
            mask = np.zeros_like(image)
        else:
            image = self.images[idx]
            mask = self.masks[idx]

        if self.augmentation:
            # Apply augmentations to numpy arrays
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Apply padding AFTER augmentations to ensure consistent padding
        image = pad_image_to_size(image, DESIRED_HEIGHT, DESIRED_WIDTH)
        mask = pad_image_to_size(mask, DESIRED_HEIGHT, DESIRED_WIDTH)

        # Normalize to 0-1 range
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Convert to tensors with channel dimension at the very end
        image = (
            torch.from_numpy(image).unsqueeze(0).float()
        )  # Add channel dim: (H, W) -> (1, H, W)
        mask = (
            torch.from_numpy(mask).unsqueeze(0).float()
        )  # Add channel dim: (H, W) -> (1, H, W)

        return {
            "image": image,
            "mask": mask,
        }


class TumorModel(pl.LightningModule):
    def __init__(
        self,
        arch,
        encoder_name,
        encoder_weights,
        in_channels,
        out_classes,
        t_max,
        learning_rate,
        dice_weight=0.5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.arch = arch
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.t_max = t_max
        self.learning_rate = learning_rate
        self.dice_weight = dice_weight

        # Create both loss functions with explicit reduction and stability parameters
        self.dice_loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE,
            from_logits=True,
            smooth=1e-6,  # Add smoothing to prevent division by zero
            eps=1e-7,  # Add epsilon for numerical stability
        )
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        return self.model(image)

    def shared_step(self, batch, stage):

        image = batch["image"]

        # Comprehensive input validation for images
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Calculate individual losses
        dice_loss = self.dice_loss_fn(logits_mask, mask)
        bce_loss = self.bce_loss_fn(logits_mask, mask)

        # Combine losses with simplex (beta and 1-beta)
        loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        # Step-level logging disabled - only logging on epochs

        return {
            "loss": loss,
            "dice_loss": dice_loss,
            "bce_loss": bce_loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        # Calculate average losses for the epoch
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_dice_loss = torch.stack([x["dice_loss"] for x in outputs]).mean()
        avg_bce_loss = torch.stack([x["bce_loss"] for x in outputs]).mean()

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_loss": avg_loss,
            f"{stage}_dice_loss": avg_dice_loss,
            f"{stage}_bce_loss": avg_bce_loss,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        # Only return the loss for PyTorch Lightning logging
        return train_loss_info["loss"]

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        # Only return the loss for PyTorch Lightning logging
        return valid_loss_info["loss"]

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        # Only return the loss for PyTorch Lightning logging
        return test_loss_info["loss"]

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        # Add eps parameter for numerical stability and weight_decay if available
        optimizer_kwargs = {
            "lr": self.learning_rate,
            "eps": 1e-8,  # Improved numerical stability
        }
        # Add weight_decay if it's set in the model
        if hasattr(self, "weight_decay"):
            optimizer_kwargs["weight_decay"] = self.weight_decay

        optimizer = torch.optim.Adam(self.parameters(), **optimizer_kwargs)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def get_train_augs() -> A.Compose:
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                shift_limit_y=0,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                interpolation=cv2.INTER_NEAREST,
                p=0.25,
                fill=255,  # was 1 in [0–1], now 255 in [0–255]
            ),
            # Flipping horizontally and vertically
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(
                alpha=10,
                sigma=100,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                approximate=True,
                p=0.2,
                fill=255,  # updated for [0–255]
            ),
            # Gaussian blur
            A.GaussianBlur(blur_limit=(3, 7), p=0.25),
            A.GaussNoise(std_range=(0.05, 0.15), p=0.25),
            A.RandomGamma(gamma_limit=(95, 105), p=0.25),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.05,
                p=0.5,
            ),
            # Removed A.Normalize() - normalization will happen after padding in CustomDataset
        ]
    )


def train(
    config,
    project_name="tumor-segmentation",
    experiment_name=None,
    enable_wandb=True,
):
    # Initialize logger
    if enable_wandb:
        logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            log_model=True,  # Log model checkpoints
            save_dir="./wandb_logs",
        )
        # Log hyperparameters
        logger.experiment.config.update(config)
    else:
        logger = TensorBoardLogger(
            save_dir="./lightning_logs",
            name=experiment_name or "default",
        )

    image_dir = "data/patients/imgs"
    mask_dir = "data/patients/labels"
    control_dir = "data/controls/imgs"

    images, masks, control = get_images(image_dir, mask_dir, control_dir)

    images, masks, control = process_data(images, masks, control, image_transform)

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    # Log dataset information
    if enable_wandb:
        logger.experiment.config.update(
            {
                "train_samples": len(train_imgs),
                "val_samples": len(val_imgs),
                "control_samples": len(control),
                "image_size": f"{DESIRED_HEIGHT}x{DESIRED_WIDTH}",
            }
        )

    augmentation = get_train_augs()

    train_dataset = CustomDataset(
        images=train_imgs,
        masks=train_masks,
        control=control,
        augmentation=augmentation,
        control_prob=config["control_prob"],
    )
    val_dataset = CustomDataset(images=val_imgs, masks=val_masks)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=4,  # Enable multiprocessing for faster data loading
        pin_memory=True,  # Faster GPU transfers
        persistent_workers=True,  # Keep workers alive between epochs
        worker_init_fn=seed_worker,  # Proper worker-level seeding
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,  # Fewer workers for validation since no augmentation
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,  # Proper worker-level seeding
    )

    # Create callbacks
    callbacks = []

    # Add early stopping callback
    early_stopping = EarlyStopping(
        monitor=config.get("early_stopping_monitor", "valid_loss"),
        min_delta=config.get("early_stopping_min_delta", 0.001),
        patience=config.get("early_stopping_patience", 7),
        verbose=True,
        mode=config.get("early_stopping_mode", "min"),
    )
    callbacks.append(early_stopping)

    # Add custom dice loss threshold early stopping
    dice_threshold_stopping = DiceLossThresholdEarlyStopping(
        threshold=config.get("dice_threshold", 0.8),
        check_epoch=config.get("dice_check_epoch", 25),
        monitor="valid_dice_loss",
    )
    callbacks.append(dice_threshold_stopping)

    trainer = Trainer(
        max_epochs=config["max_epochs"],
        logger=logger,
        enable_checkpointing=False,
        callbacks=callbacks,
        precision=32,  # Disable AMP to avoid gradient scaler issues
        gradient_clip_val=1.0,  # Add gradient clipping to prevent exploding gradients
        gradient_clip_algorithm="norm",
    )

    model = TumorModel(
        arch=config["architecture"],
        encoder_name=config["encoder"],
        encoder_weights=config["encoder_weights"],
        in_channels=1,
        out_classes=1,
        learning_rate=config["learning_rate"],
        t_max=config["max_epochs"] * len(train_loader),
        dice_weight=config.get("dice_weight", 0.5),
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save the model after training is complete
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # Save the complete model (PyTorch Lightning checkpoint)
    model_path = os.path.join(
        save_dir, f"final_model_{experiment_name or 'default'}.ckpt"
    )
    trainer.save_checkpoint(model_path)
    print(f"Final model saved to: {model_path}")

    # Also save just the model state dict for easier loading
    state_dict_path = os.path.join(
        save_dir, f"final_model_state_dict_{experiment_name or 'default'}.pth"
    )
    torch.save(model.state_dict(), state_dict_path)
    print(f"Model state dict saved to: {state_dict_path}")

    # Log the model save paths if using wandb
    if enable_wandb and hasattr(logger, "experiment"):
        logger.experiment.log_model(model_path)


def run_experiment(
    config, project_name="tumor-segmentation", experiment_name=None, enable_wandb=True
):
    """Run a single experiment with the given configuration"""
    try:
        train(
            config,
            project_name=project_name,
            experiment_name=experiment_name,
            enable_wandb=enable_wandb,
        )
    except Exception as e:
        print(f"Error in run_experiment: {e}")
        raise e
    finally:
        if enable_wandb:
            wandb.finish()


if __name__ == "__main__":

    # Run single experiment with great params
    config = {
        "learning_rate": 0.0001,
        "control_prob": 0,
        "max_epochs": 50,  # Increased to allow for the 25-epoch check
        "batch_size": 2,
        "architecture": "unetplusplus",
        "encoder": "efficientnet-b0",
        "encoder_weights": "imagenet",
        "loss_function": "DiceLoss+BCE",
        "dice_weight": 0.25,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        # Early stopping parameters (optional)
        "early_stopping_monitor": "valid_dice_loss",  # can also use "valid_dataset_iou"
        "early_stopping_patience": 7,  # number of epochs to wait for improvement
        "early_stopping_min_delta": 0.001,  # minimum change to qualify as improvement
        "early_stopping_mode": "min",  # "min" for loss, "max" for accuracy/IoU
        # Dice loss threshold early stopping parameters
        "dice_threshold": 0.7,  # stop if validation dice loss is not below this value
        "dice_check_epoch": 25,  # check the threshold after this many epochs
    }

    # To disable wandb, set enable_wandb=False
    # This will use TensorBoard logging instead
    run_experiment(config, experiment_name="baseline_experiment", enable_wandb=False)

    # Example of running without wandb:
    # run_experiment(config, experiment_name="baseline_experiment_no_wandb", enable_wandb=False)

    # Uncomment the line below to run hyperparameter sweep
    # run_hyperparameter_sweep()
