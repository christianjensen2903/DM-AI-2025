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
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pytorch_lightning.callbacks import Callback

DESIRED_WIDTH = 416
DESIRED_HEIGHT = 992


class WandbImageCallback(Callback):
    """Custom callback to log sample images to wandb during training"""

    def __init__(self, log_frequency=5, max_samples=4, enable_wandb=True):
        self.log_frequency = log_frequency
        self.max_samples = max_samples
        self.enable_wandb = enable_wandb

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            self.enable_wandb
            and trainer.current_epoch % self.log_frequency == 0
            and hasattr(trainer.logger, "experiment")
        ):
            # Get a batch from validation dataloader
            val_loader = trainer.val_dataloaders
            batch = next(iter(val_loader))

            # Move batch to the same device as model
            device = pl_module.device
            batch = {k: v.to(device) for k, v in batch.items()}

            pl_module.log_sample_images(
                batch, stage="val", max_samples=self.max_samples
            )


def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor, eps=1e-6):
    y_true = y_true > 0.5
    y_pred = y_pred > 0.5
    inter = (y_true & y_pred).sum().float()
    denom = y_true.sum().float() + y_pred.sum().float()
    return (2 * inter / (denom + eps)).item()


def normalize_for_display(tensor):
    # tensor: 2D torch float tensor
    t = tensor.clone().float()
    t = t - t.min()
    maxv = t.max()
    if maxv > 0:
        t = t / maxv
    # return numpy 0..255 uint8
    return (t * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


def read_image_as_numpy(path):
    """Load image as numpy array with shape (H, W)"""
    image = Image.open(path).convert("L")  # Convert to grayscale
    return np.array(image)


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


def image_transform_numpy(image):
    """Transform numpy image to desired size and normalize"""
    height, width = image.shape

    # Calculate padding needed
    bottom_pad = max(0, DESIRED_HEIGHT - height)
    left_pad = max(0, int(np.floor((DESIRED_WIDTH - width) / 2)))
    right_pad = max(0, int(np.ceil((DESIRED_WIDTH - width) / 2)))

    # Apply padding
    image = np.pad(
        image,
        ((0, bottom_pad), (left_pad, right_pad)),
        mode="constant",
        constant_values=0,
    )

    # Crop if image is larger than desired size
    image = image[:DESIRED_HEIGHT, :DESIRED_WIDTH]

    # Normalize to 0-1 range
    image = image.astype(np.float32) / 255.0

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
            random_seed = random.randint(0, 2**32)
            torch.manual_seed(random_seed)
            random.seed(random_seed)

            # Apply augmentations to numpy arrays
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

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
        beta=0.5,
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
        self.beta = beta

        # Create both loss functions
        self.dice_loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True
        )
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        return self.model(image)

    def shared_step(self, batch, stage):

        image = batch["image"]

        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"].clamp(0, 1)
        assert mask.ndim == 4
        assert mask.max() <= 1 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Calculate individual losses
        dice_loss = self.dice_loss_fn(logits_mask, mask)
        bce_loss = self.bce_loss_fn(logits_mask, mask)

        # Combine losses with simplex (beta and 1-beta)
        loss = self.beta * dice_loss + (1 - self.beta) * bce_loss

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )

        # Log losses for each step
        self.log(
            f"{stage}_loss_step", loss, on_step=True, on_epoch=False, prog_bar=False
        )
        self.log(
            f"{stage}_dice_loss_step",
            dice_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log(
            f"{stage}_bce_loss_step",
            bce_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

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
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def log_sample_images(self, batch, stage="val", max_samples=4):
        """Log one composite figure per sample: image, GT, pred, TP/FP/FN overlay + dice"""
        if not hasattr(self.logger, "experiment") or not isinstance(
            self.logger, WandbLogger
        ):
            return

        images = batch["image"][:max_samples]  # expected shape (B, 1, H, W)
        masks = batch["mask"][:max_samples]  # expected shape (B, 1, H, W)

        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.sigmoid(logits)
            preds = probs > 0.5  # bool mask

        wandb_images = []
        for i in range(images.shape[0]):
            img = images[i, 0]  # (H, W)
            gt = masks[i, 0]  # (H, W)
            pred = preds[i, 0]  # (H, W) bool

            # Dice
            dice = dice_coef(gt, pred)

            # Normalize image for display
            img_disp = normalize_for_display(img)  # uint8 HxW
            gt_disp = (gt > 0.5).to(torch.uint8).cpu().numpy() * 255
            pred_disp = pred.to(torch.uint8).cpu().numpy() * 255

            # TP / FP / FN
            gt_bool = gt > 0.5
            pred_bool = pred
            tp = pred_bool & gt_bool
            fp = pred_bool & ~gt_bool
            fn = ~pred_bool & gt_bool

            # Build overlay RGB: start with grayscale image as background
            background = (
                np.stack([img_disp] * 3, axis=-1).astype(float) / 255.0
            )  # normalized 0..1

            overlay = np.zeros_like(background)  # float
            # green=TP, red=FP, blue=FN
            overlay[..., 1][tp.cpu().numpy()] = 1.0  # TP
            overlay[..., 0][fp.cpu().numpy()] = 1.0  # FP
            overlay[..., 2][fn.cpu().numpy()] = 1.0  # FN

            # Alpha blend overlay on background so you still see anatomy
            alpha = 0.6
            blended = np.clip((1 - alpha) * background + alpha * overlay, 0, 1)

            # Composite figure
            fig, axs = plt.subplots(1, 4, figsize=(12, 3.5))
            axs[0].imshow(img_disp, cmap="gray")
            axs[0].set_title("Input Image")
            axs[0].axis("off")

            axs[1].imshow(gt_disp, cmap="gray")
            axs[1].set_title("Ground Truth")
            axs[1].axis("off")

            axs[2].imshow(pred_disp, cmap="gray")
            axs[2].set_title("Prediction")
            axs[2].axis("off")

            axs[3].imshow(blended)
            axs[3].set_title(f"Overlay (dice={dice:.2f})")
            axs[3].axis("off")

            # Legend (single legend on last panel)
            legend_elements = [
                Patch(facecolor="green", edgecolor="black", label="TP"),
                Patch(facecolor="red", edgecolor="black", label="FP"),
                Patch(facecolor="blue", edgecolor="black", label="FN"),
            ]
            axs[3].legend(handles=legend_elements, loc="lower right", framealpha=0.9)

            plt.tight_layout()

            wandb_images.append(
                wandb.Image(fig, caption=f"{stage}_sample_{i} dice={dice:.2f}")
            )
            plt.close(fig)

        # Log them as a batch list
        self.logger.experiment.log({f"{stage}_samples": wandb_images})

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=5,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.75,
                fill=1,
            ),
            A.ElasticTransform(
                alpha=5,
                sigma=50,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                approximate=True,
                p=0.20,
                fill=1,
            ),
            # ── Intensity ───────────────────────────────────────────────
            A.RandomGamma(gamma_limit=(95, 105), p=0.25),
            A.RandomBrightnessContrast(
                brightness_limit=0.05, contrast_limit=0.10, p=0.5
            ),
        ],
        additional_targets={"mask": "mask"},
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

    images, masks, control = process_data(images, masks, control, image_transform_numpy)

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
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Create callbacks
    callbacks = []
    if enable_wandb:
        image_callback = WandbImageCallback(
            log_frequency=2, max_samples=4, enable_wandb=True
        )
        callbacks.append(image_callback)

    trainer = Trainer(
        max_epochs=config["max_epochs"],
        log_every_n_steps=1,
        logger=logger,
        enable_checkpointing=True,
        callbacks=callbacks,
    )

    model = TumorModel(
        arch=config["architecture"],
        encoder_name=config["encoder"],
        encoder_weights=config["encoder_weights"],
        in_channels=1,
        out_classes=1,
        learning_rate=config["learning_rate"],
        t_max=config["max_epochs"] * len(train_loader),
        beta=config.get("beta", 0.5),
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Log final model performance
    if enable_wandb:
        logger.experiment.finish()


def run_experiment(
    config, project_name="tumor-segmentation", experiment_name=None, enable_wandb=True
):
    """Run a single experiment with the given configuration"""
    train(
        config,
        project_name=project_name,
        experiment_name=experiment_name,
        enable_wandb=enable_wandb,
    )


def run_hyperparameter_sweep(enable_wandb=True):
    """Example function to run multiple experiments with different hyperparameters"""
    configs = [
        {
            "learning_rate": 0.0001,
            "control_prob": 0.2,
            "dropout": 0.2,
            "max_epochs": 10,
        },
        {
            "learning_rate": 0.0002,
            "control_prob": 0.3,
            "dropout": 0.1,
            "max_epochs": 10,
        },
        {
            "learning_rate": 0.00005,
            "control_prob": 0.1,
            "dropout": 0.3,
            "max_epochs": 10,
        },
    ]

    for i, config in enumerate(configs):
        experiment_name = f"experiment_{i+1}_lr{config['learning_rate']}_cp{config['control_prob']}_do{config['dropout']}"
        print(f"Running {experiment_name}")
        run_experiment(
            config, experiment_name=experiment_name, enable_wandb=enable_wandb
        )


if __name__ == "__main__":
    # Run single experiment with great params
    config = {
        "learning_rate": 0.0001,
        "control_prob": 0,
        "max_epochs": 10,
        "batch_size": 2,
        "architecture": "unetplusplus",
        "encoder": "efficientnet-b0",
        "encoder_weights": "imagenet",
        "loss_function": "DiceLoss+BCE",
        "beta": 0.25,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
    }

    # To disable wandb, set enable_wandb=False
    # This will use TensorBoard logging instead
    run_experiment(config, experiment_name="baseline_experiment", enable_wandb=False)

    # Example of running without wandb:
    # run_experiment(config, experiment_name="baseline_experiment_no_wandb", enable_wandb=False)

    # Uncomment the line below to run hyperparameter sweep
    # run_hyperparameter_sweep()
