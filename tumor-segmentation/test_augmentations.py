import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import albumentations as A


def load_sample_images(num_samples=3):
    """Load sample images from the dataset"""
    patient_dir = "data/patients/imgs"
    mask_dir = "data/patients/labels"

    sample_images = {}

    # Load patient images
    if os.path.exists(patient_dir):
        patient_files = [f for f in os.listdir(patient_dir) if f.endswith(".png")][
            :num_samples
        ]
        sample_images["patients"] = []
        sample_images["masks"] = []

        for file in patient_files:
            # Load patient image
            img_path = os.path.join(patient_dir, file)
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img_array = np.array(img) / 255.0  # Normalize to 0-1 range
            sample_images["patients"].append(img_array)

            # Load corresponding mask
            mask_file = file.replace("patient", "segmentation")
            mask_path = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                mask_array = np.array(mask) / 255.0  # Normalize to 0-1 range
                sample_images["masks"].append(mask_array)
            else:
                sample_images["masks"].append(np.zeros_like(img_array))

    return sample_images


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


def main():

    n = 5

    sample_images = load_sample_images(n)

    fig, axes = plt.subplots(n, 2, figsize=(12, 12))

    for i, (patient_img, mask_img) in enumerate(
        zip(sample_images["patients"][:n], sample_images["masks"][:n])
    ):
        # Apply multiple augmentations
        random.seed(42 + i)
        np.random.seed(42 + i)

        augmentations = get_train_augs()

        # Combined augmentation pipeline
        augmented_img = patient_img.copy()
        augmented_mask = mask_img.copy()

        augmented = augmentations(image=augmented_img, mask=augmented_mask)
        augmented_img = augmented["image"]
        augmented_mask = augmented["mask"]

        # Show original and augmented
        row = i
        axes[row, 0].imshow(patient_img, cmap="gray")
        axes[row, 0].set_title(f"Original Patient {i+1}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(augmented_img, cmap="gray")
        axes[row, 1].set_title(f"Augmented Patient {i+1}")
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
