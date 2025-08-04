#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import nibabel as nib
import json
from tqdm import tqdm


def read_grayscale_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return arr


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    # Any nonzero is considered foreground; output 0/1 integer mask
    bin_mask = (mask > 0).astype(np.uint8)
    return bin_mask


def save_nifti(volume: np.ndarray, out_path: Path):
    # nnU-Net expects data in (X,Y,Z). For 2D images we add a singleton third axis.
    if volume.ndim == 2:
        volume = volume[:, :, None]
    # Use identity affine; spacing is unspecified (nnU-Net will infer/default)
    nif = nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4))
    nib.save(nif, str(out_path))


def main():
    p = argparse.ArgumentParser(
        description="Convert custom dataset to nnU-Net raw format (v2)."
    )
    p.add_argument(
        "--image_dir",
        type=Path,
        default=Path("data/patients/imgs"),
        help="Directory with input images.",
    )
    p.add_argument(
        "--mask_dir",
        type=Path,
        default=Path("data/patients/labels"),
        help="Directory with ground truth masks.",
    )
    p.add_argument(
        "--control_dir",
        type=Path,
        default=Path("data/controls/imgs"),
        help="Directory with control images (optional).",
    )
    p.add_argument(
        "--output_base",
        type=Path,
        default=Path("nnUNet_raw"),
        help="Base output folder (will contain DatasetXXX_Name).",
    )
    p.add_argument(
        "--task_id", type=int, default=500, help="Three-digit dataset ID (e.g. 500)."
    )
    p.add_argument(
        "--task_name",
        type=str,
        default="Tumor",
        help="Descriptive name for the dataset.",
    )
    p.add_argument(
        "--include_controls",
        action="store_true",
        help="Include control images as negative examples (zero mask).",
    )
    args = p.parse_args()

    # Build dataset folder
    dataset_folder = args.output_base / f"Dataset{args.task_id:03d}_{args.task_name}"
    imagesTr = dataset_folder / "imagesTr"
    labelsTr = dataset_folder / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    training_entries = []

    # Process patient image-mask pairs
    img_files = sorted([f for f in os.listdir(args.image_dir) if not f.startswith(".")])
    for img_fname in tqdm(img_files, desc="Processing patient cases"):
        img_path = args.image_dir / img_fname
        # Derive mask name by same logic as existing code
        mask_fname = img_fname.replace("patient", "segmentation")
        mask_path = args.mask_dir / mask_fname
        if not mask_path.exists():
            print(
                f"Warning: mask {mask_fname} not found for image {img_fname}, skipping."
            )
            continue

        case_id = Path(img_fname).stem  # e.g., patient001
        img_arr = read_grayscale_image(img_path)  # float32
        mask_arr = read_grayscale_image(mask_path)
        mask_arr = ensure_binary_mask(mask_arr)

        # Save image and label in nnUNet naming convention
        image_out_name = f"{case_id}_0000.nii.gz"
        label_out_name = f"{case_id}.nii.gz"

        save_nifti(img_arr, imagesTr / image_out_name)
        save_nifti(mask_arr, labelsTr / label_out_name)

        training_entries.append(
            {
                "image": f"./imagesTr/{image_out_name}",
                "label": f"./labelsTr/{label_out_name}",
            }
        )

    # Optionally add controls as negative examples
    if args.include_controls and args.control_dir.exists():
        control_files = sorted(
            [f for f in os.listdir(args.control_dir) if not f.startswith(".")]
        )
        for idx, ctrl_fname in enumerate(
            tqdm(control_files, desc="Processing controls")
        ):
            ctrl_path = args.control_dir / ctrl_fname
            case_id = f"control{idx:03d}"
            img_arr = read_grayscale_image(ctrl_path)
            mask_arr = np.zeros_like(img_arr, dtype=np.uint8)  # zero mask

            image_out_name = f"{case_id}_0000.nii.gz"
            label_out_name = f"{case_id}.nii.gz"
            save_nifti(img_arr, imagesTr / image_out_name)
            save_nifti(mask_arr, labelsTr / label_out_name)

            training_entries.append(
                {
                    "image": f"./imagesTr/{image_out_name}",
                    "label": f"./labelsTr/{label_out_name}",
                }
            )

    if len(training_entries) == 0:
        raise RuntimeError(
            "No training cases were written. Check your input directories and naming conventions."
        )

    # Build dataset.json for nnU-Net v2 (simplified format)
    dataset_json = {
        "channel_names": {
            "0": "CT"  # adjust if modality is different; "CT" triggers global intensity norm
        },
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(training_entries),
        "file_ending": ".nii.gz",
    }

    # Write training list into a separate file as nnUNet expects training info via the list in code;
    # For v2, nnUNet will auto-generate splits from dataset.json + folder contents.
    # The original dataset.json format does not embed the list of cases for v2 (unlike v1), so we save an auxiliary train list if desired.
    # However, many examples still keep a 'training' list; to be safe, include it here:
    dataset_json["training"] = training_entries

    with open(dataset_folder / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Finished. Dataset written to {dataset_folder}")
    print(f"Total training cases (including controls if any): {len(training_entries)}")
    print("Next steps:")
    print(
        f"  1. Export environment variables for nnU-Net, e.g.:\n"
        f'     export nnUNet_raw_data_base="{args.output_base.parent / args.output_base.name}_base"\n'
        f'     export nnUNet_preprocessed="<your preprocess output>"\n'
        f'     export RESULTS_FOLDER="<your trained models>"'
    )
    print("  2. Run nnUNet planning & preprocessing, e.g.:")
    print(f"     nnUNetv2_extract_fingerprint -d {args.task_id}")
    print(f"     nnUNetv2_plan_experiment -d {args.task_id}")
    print(
        f"     nnUNetv2_preprocess -d {args.task_id} -c 2d"
    )  # or appropriate configuration


if __name__ == "__main__":
    main()
