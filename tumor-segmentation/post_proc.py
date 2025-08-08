# sweep_threshold_mincomp.py
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    binary_opening,
    disk,
)
from sklearn.model_selection import train_test_split
import pandas as pd

# Import from your existing codebase
from main import (
    TumorModel,
    read_image_as_numpy,
    pad_image_to_size,
    DESIRED_WIDTH,
    DESIRED_HEIGHT,
)


# ----------------------------
# Data helpers
# ----------------------------
def listdir_sorted(p):
    # ensures deterministic pairing
    return sorted([f for f in os.listdir(p) if not f.startswith(".")])


def load_patients(image_dir, mask_dir):
    imgs, gts, img_names = [], [], []
    image_fns = listdir_sorted(image_dir)
    for img_fn in image_fns:
        mask_fn = img_fn.replace("patient", "segmentation")
        img_path = os.path.join(image_dir, img_fn)
        mask_path = os.path.join(mask_dir, mask_fn)
        if not os.path.exists(mask_path):
            continue
        img = read_image_as_numpy(img_path)  # HxW uint8
        msk = read_image_as_numpy(mask_path)  # HxW uint8
        imgs.append(img)
        gts.append(msk)
        img_names.append(img_fn)
    return imgs, gts, img_names


def crop_back_from_pad(padded, h_orig, w_orig):
    """Inverse of pad_image_to_size for center horizontal padding and bottom padding."""
    h_pad, w_pad = padded.shape
    bottom_pad = max(0, DESIRED_HEIGHT - h_orig)
    left_pad = int(np.floor((DESIRED_WIDTH - w_orig) / 2))
    right_pad = int(np.ceil((DESIRED_WIDTH - w_orig) / 2))
    # slice H first (remove bottom pad), then W (remove left/right pads)
    cropped = padded[: h_pad - bottom_pad, left_pad : w_pad - right_pad]
    # In case of any off-by-one, enforce exact size
    return cropped[:h_orig, :w_orig]


# ----------------------------
# Model / inference helpers
# ----------------------------
def load_models(ckpt_paths, device):
    models = []
    for p in ckpt_paths:
        m = TumorModel.load_from_checkpoint(p, map_location=device)
        m.eval()
        m.to(device)
        models.append(m)
    return models


@torch.no_grad()
def predict_ensemble_prob(img_np, models, device, hflip_tta=False):
    """
    img_np: HxW uint8 (single-channel)
    returns: probability map as float32 numpy array on padded canvas (DESIRED_HEIGHT x DESIRED_WIDTH)
    """
    # pad like your serving code
    padded = pad_image_to_size(img_np, DESIRED_HEIGHT, DESIRED_WIDTH)  # HxW uint8
    x = (
        torch.from_numpy(padded).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
    )  # [1,1,H,W]

    logits = None
    for m in models:
        out = m.forward(x)  # [1,1,H,W] logits
        logits = out if logits is None else (logits + out)
    logits = logits / float(len(models))

    if hflip_tta:
        x_flip = torch.flip(x, dims=[3])
        logits_flip = None
        for m in models:
            outf = m.forward(x_flip)
            logits_flip = outf if logits_flip is None else (logits_flip + outf)
        logits_flip = logits_flip / float(len(models))
        logits_flip = torch.flip(logits_flip, dims=[3])
        logits = 0.5 * (logits + logits_flip)

    probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().float().numpy()  # HxW
    return probs


def dice_score_bin(pred_bin, gt_bin, eps=1e-7):
    # pred_bin, gt_bin: HxW uint8/bool {0,1}
    inter = np.logical_and(pred_bin, gt_bin).sum(dtype=np.float64)
    s = pred_bin.sum(dtype=np.float64) + gt_bin.sum(dtype=np.float64)
    return (2.0 * inter + eps) / (s + eps)


def postprocess_binary(mask_bool, min_size, morph):
    if min_size > 0:
        mask_bool = remove_small_objects(mask_bool, min_size=min_size)
    if morph:
        mask_bool = binary_closing(mask_bool, disk(1))
        mask_bool = binary_opening(mask_bool, disk(1))
    return mask_bool


# ----------------------------
# Main sweep
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Threshold & min-component sweep for Dice."
    )
    parser.add_argument(
        "--ckpt", nargs="+", required=True, help="One or more .ckpt paths (ensemble)."
    )
    parser.add_argument("--image_dir", default="data/patients/imgs")
    parser.add_argument("--mask_dir", default="data/patients/labels")
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thr_start", type=float, default=0.25)
    parser.add_argument("--thr_end", type=float, default=0.60)
    parser.add_argument("--thr_step", type=float, default=0.05)
    parser.add_argument(
        "--min_sizes", nargs="*", type=int, default=[0, 10, 25, 50, 100, 200]
    )
    parser.add_argument(
        "--hflip_tta", action="store_true", help="Use horizontal flip TTA."
    )
    parser.add_argument(
        "--morph",
        action="store_true",
        help="Apply light closing->opening after small-object removal.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out_csv", default="sweep_results.csv")
    args = parser.parse_args()

    # Load data
    imgs, gts, names = load_patients(args.image_dir, args.mask_dir)
    idx_train, idx_val = train_test_split(
        np.arange(len(imgs)),
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
    )
    val_idx = sorted(idx_val.tolist())

    # Build validation lists
    val_imgs = [imgs[i] for i in val_idx]
    val_gts = [gts[i] for i in val_idx]
    val_names = [names[i] for i in val_idx]

    # Load models
    device = torch.device(args.device)
    models = load_models(args.ckpt, device)

    # Precompute probabilities on padded canvas for all val images (so we sweep fast)
    print(
        f"Inference on {len(val_imgs)} validation images (device={device.type}, models={len(models)})..."
    )
    probs_list = []
    sizes_list = []
    for im in tqdm(val_imgs):
        H, W = im.shape
        probs = predict_ensemble_prob(
            im, models, device, hflip_tta=args.hflip_tta
        )  # DESIRED_HxDESIRED_W
        probs_list.append(probs.astype(np.float32))
        sizes_list.append((H, W))

    # Prepare thresholds
    thresholds = np.arange(args.thr_start, args.thr_end + 1e-9, args.thr_step)

    # Sweep
    rows = []
    best = {"dice": -1.0, "thr": None, "min_size": None}
    print("Sweeping thresholds and min-component sizes...")
    for thr in thresholds:
        for min_sz in args.min_sizes:
            dices = []
            for probs, (h0, w0), gt in zip(probs_list, sizes_list, val_gts):
                # Binarize on padded canvas
                pred_bool = probs > thr
                # Post-proc
                pred_bool = postprocess_binary(pred_bool, min_sz, args.morph)
                # Crop back to original size for fair comparison
                pred_cropped = crop_back_from_pad(pred_bool, h0, w0)
                # Prepare GT as bool (assumes GT in {0,255} or {0,1})
                gt_bool = (gt > 0).astype(bool)
                # Dice
                d = dice_score_bin(pred_cropped, gt_bool)
                dices.append(d)
            mean_dice = float(np.mean(dices))
            rows.append({"threshold": thr, "min_size": min_sz, "dice": mean_dice})
            if mean_dice > best["dice"]:
                best = {"dice": mean_dice, "thr": float(thr), "min_size": int(min_sz)}

    df = pd.DataFrame(rows).sort_values(["dice"], ascending=False)
    df.to_csv(args.out_csv, index=False)

    print("\n===== BEST =====")
    print(
        f"Dice: {best['dice']:.4f} @ threshold={best['thr']:.2f}, min_size={best['min_size']}"
    )
    print(f"Full results saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
