import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
import albumentations as A

# Constants for desired image size
DESIRED_WIDTH = 416
DESIRED_HEIGHT = 992


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
            img_array = np.array(img)
            sample_images["patients"].append(img_array)

            # Load corresponding mask
            mask_file = file.replace("patient", "segmentation")
            mask_path = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                mask_array = np.array(mask)
                sample_images["masks"].append(mask_array)
            else:
                sample_images["masks"].append(np.zeros_like(img_array))

    return sample_images


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
            A.RandomGamma(gamma_limit=(95, 105), p=1),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.05,
                p=1,
            ),
            # A.Normalize(),
            # Removed A.Normalize() - normalization will happen after padding in CustomDataset
        ]
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

        # Apply padding AFTER augmentations to ensure consistent padding
        augmented_img = pad_image_to_size(augmented_img, DESIRED_HEIGHT, DESIRED_WIDTH)
        augmented_mask = pad_image_to_size(
            augmented_mask, DESIRED_HEIGHT, DESIRED_WIDTH
        )

        # Check if mask is binary and has only 0 and 1
        print(augmented_img.min(), augmented_img.max())

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
