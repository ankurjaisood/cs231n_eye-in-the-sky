#!/usr/bin/env python3

import os
import sys
import torch
import argparse
import datetime
import wandb
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.optim import AdamW
import albumentations as A

 # 52 cards + background
id2label = {
    0:  "background",
    1:  "A_C",  2: "2_C",  3: "3_C",  4: "4_C",  5: "5_C",  6: "6_C",  7: "7_C",  8: "8_C",  9: "9_C", 10: "10_C", 11: "J_C",  12:"Q_C", 13: "K_C",
    14: "A_S", 15: "2_S", 16: "3_S", 17: "4_S", 18: "5_S", 19: "6_S", 20: "7_S", 21: "8_S", 22: "9_S", 23: "10_S", 24: "J_S", 25: "Q_S", 26: "K_S",
    27: "A_H", 28: "2_H", 29: "3_H", 30: "4_H", 31: "5_H", 32: "6_H", 33: "7_H", 34: "8_H", 35: "9_H", 36: "10_H", 37: "J_H", 38: "Q_H", 39: "K_H",
    40: "A_D", 41: "2_D", 42: "3_D", 43: "4_D", 44: "5_D", 45: "6_D", 46: "7_D", 47: "8_D", 48: "9_D", 49: "10_D", 50: "J_D", 51: "Q_D", 52: "K_D",
}

# High, Low, None for each suit
id2label_suits_category = {
    0:  "background",
    1:  "C_Low",  2: "C_None", 3:  "C_High",
    4:  "S_Low",  5: "S_None", 6:  "S_High",
    7:  "H_Low",  8: "H_None", 9:  "H_High",
    10: "D_Low", 11: "D_None", 12: "D_High",
}

# High, Low, None
id2label_category = {
    0:  "background",
    1:  "Low", 
    2:  "None", 
    3:  "High",
}

class SegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, feature_extractor, size=512, augmentations=None):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.extractor  = feature_extractor
        self.size       = size
        self.augmentations = augmentations

        # List all image filenames
        imgs = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        # For each image, assume a mask with the same basename exists
        self.pairs = []
        for img_fname in imgs:
            base, ext = os.path.splitext(img_fname)
            img_path  = os.path.join(images_dir, img_fname)
            mask_path = os.path.join(masks_dir, base + ext)

            # Assert that the corresponding mask file actually exists
            assert os.path.exists(mask_path), f"Mask not found for image {img_path}: expected {mask_path}"
            self.pairs.append((img_path, mask_path))
        
        assert len(self.pairs) > 0, f"No image/mask pairs found in {images_dir} and {masks_dir}"
        assert len(self.pairs) == len(imgs), "Mismatch between number of images and masks"
        assert len(self.pairs) == len(imgs), "Duplicate image filenames found"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, msk_path = self.pairs[i]

        # Load image at its native size 
        img = Image.open(img_path).convert("RGB")

        # Load mask at its native size
        msk = Image.open(msk_path).convert("L")

        img_np  = np.array(img)
        msk_np  = np.array(msk) 

        if self.augmentations is not None:
            augmented = self.augmentations(image=img_np, mask=msk_np)
            img_np = augmented["image"]
            msk_np = augmented["mask"]
        img_aug = Image.fromarray(img_np)

        # Use the feature extractor to: resize to self.size, normalize, convert to tensor.
        encoding = self.extractor(
            images=img_aug,
            size=self.size,            # both height and width → self.size (e.g. 512)
            return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze(0)  # shape: (3, self.size, self.size)

        # Resize the mask the same way, with nearest neighbor (to preserve integer labels).
        msk_resized = Image.fromarray(msk_np).resize((self.size, self.size), resample=Image.NEAREST)
        labels = torch.from_numpy(np.array(msk_resized, dtype=np.int64)).long()  # (size, size)

        return pixel_values, labels

def train_validate_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not wandb.login(relogin=False):
        print("Error: W&B is not logged in. Please run `wandb login` or set WANDB_API_KEY before continuing.")
        sys.exit(1)

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run_name = f"segformer_{now_str}_{args.pretrained.split('/')[-1]}_nl-{args.num_labels}_bs-{args.batch_size}_lr-{args.lr:.0e}_epochs-{args.epochs}_is-{args.img_size}"
    wandb.init(
        project=args.wandb_project,
        name = wandb_run_name,
        config={
            "pretrained": args.pretrained,
            "num_labels": args.num_labels,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "ignore_background": args.ign_background
        }
    )
    config = wandb.config

    # Load feature extractor + model
    extractor = SegformerImageProcessor.from_pretrained(args.pretrained)
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.pretrained,
        num_labels=args.num_labels,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    if args.ign_background:
        print("Ignoring background pixels in loss calculation")
        model.config.ignore_index = 0  # Set ignore_index to 0 (background) for loss calculation

    train_augmentations = A.Compose([
        A.RandomScale(scale_limit=0.2,  # random factor in [1−0.2, 1+0.2]
                  interpolation=cv2.INTER_LINEAR,
                  mask_interpolation_method=cv2.INTER_NEAREST,
                  p=1.0),

        # Since RandomScale changes the image size, we force‐pad/crop back to a square of size 512×512
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
        A.RandomCrop(height=512, width=512),  # in case RandomScale made it bigger

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ], additional_targets={"mask": "mask"})

    # Prepare Datasets and DataLoaders
    train_ds = SegDataset(
        images_dir=args.train_images,
        masks_dir=args.train_masks,
        feature_extractor=extractor,
        size=args.img_size,
        augmentations=train_augmentations,
    )
    valid_ds = SegDataset(
        images_dir=args.valid_images,
        masks_dir=args.valid_masks,
        feature_extractor=extractor,
        size=args.img_size,
    )
    test_ds = SegDataset(
        images_dir=args.test_images,
        masks_dir=args.test_masks,
        feature_extractor=extractor,
        size=args.img_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.wd
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_correct = 0
        train_total   = 0
        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(device)  # (B, 3, H, W)
            labels       = labels.to(device)        # (B, H, W)
            outputs = model(pixel_values=pixel_values, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_accum += loss.item()

            with torch.no_grad():
                logits = outputs.logits            # (B, C, H, W)
                logits_up = torch.nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],  # (H_label, W_label)
                    mode="bilinear",
                    align_corners=False
                )
                preds  = torch.argmax(logits_up, dim=1)  # (B, H, W)

                if config.ignore_background:
                    mask = (labels != 0)
                    train_correct += (preds[mask] == labels[mask]).sum().item()
                    train_total   += mask.sum().item()
                else:
                    train_correct += (preds == labels).sum().item()
                    train_total   += preds.numel()

        avg_train_loss = train_loss_accum / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        wandb.log({
            "train_loss": avg_train_loss,
            "train_accuracy": train_acc
        }, step=epoch)
        print(f"Epoch {epoch}/{config.epochs} — Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        valid_loss_accum = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for pixel_values, labels in valid_loader:
                pixel_values = pixel_values.to(device)
                labels       = labels.to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                valid_loss_accum += outputs.loss.item()

                logits = outputs.logits  # (B, C, H, W)
                logits_up = torch.nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],  # (H_label, W_label)
                    mode="bilinear",
                    align_corners=False
                )
                preds = torch.argmax(logits_up, dim=1)  # (B, H, W)

                if config.ignore_background:
                    mask = (labels != 0)
                    preds_flat = preds[mask].cpu().numpy().flatten()
                    labels_flat = labels[mask].cpu().numpy().flatten()
                else:
                    preds_flat = preds.cpu().numpy().flatten()
                    labels_flat = labels.cpu().numpy().flatten()

                all_preds.append(preds_flat)
                all_labels.append(labels_flat)

        avg_valid_loss = valid_loss_accum / len(valid_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        valid_acc = (all_preds == all_labels).mean()
        
        wandb.log({
            "valid_loss": avg_valid_loss,
            "valid_accuracy": valid_acc
        }, step=epoch)
        print(f"Epoch {epoch}/{config.epochs} — Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

    # Final Evaluation
    model.eval()
    test_loss_accum = 0.0
    all_preds_test = []
    all_labels_test = []
    with torch.no_grad():
        for pixel_values, labels in test_loader:
            pixel_values = pixel_values.to(device)
            labels       = labels.to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            test_loss_accum += outputs.loss.item()

            logits = outputs.logits
            logits_up = torch.nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            preds = torch.argmax(logits_up, dim=1)

            if config.ignore_background:
                mask = (labels != 0)
                preds_flat = preds[mask].cpu().numpy().flatten()
                labels_flat = labels[mask].cpu().numpy().flatten()
            else:
                preds_flat = preds.cpu().numpy().flatten()
                labels_flat = labels.cpu().numpy().flatten()

            all_preds_test.append(preds_flat)
            all_labels_test.append(labels_flat)

    avg_test_loss = test_loss_accum / len(test_loader)
    all_preds_test = np.concatenate(all_preds_test)
    all_labels_test = np.concatenate(all_labels_test)
    test_acc = (all_preds_test == all_labels_test).mean()
    
    wandb.log({
        "test_loss": avg_test_loss,
        "test_accuracy": test_acc
    })
    print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Generate confusion matrix
    cm_test = confusion_matrix(
        all_labels_test, 
        all_preds_test, 
        labels=list(range(config.num_labels))
    )
    if config.ignore_background:
        cm_test[0, :] = 0
        cm_test[:, 0] = 0

    class_names = []
    if config.num_labels == 53:
        class_names = [id2label[i] for i in range(config.num_labels)]
    elif config.num_labels == 13:
        class_names = [id2label_suits_category[i] for i in range(config.num_labels)]
    elif config.num_labels == 4:
        class_names = [id2label_category[i] for i in range(config.num_labels)]
    else:
        raise ValueError(f"Unsupported number of labels: {config.num_labels}")
    
    wandb.log({
        "confusion_matrix/test":
            wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels_test,
                preds=all_preds_test,
                class_names=class_names
            )
    })

    wandb.log({
    "confusion_matrix_heatmap": wandb.sklearn.plot_confusion_matrix(
        all_labels_test,        # ground‐truth labels
        all_preds_test,         # predicted labels
        labels=list(range(config.num_labels))
        )
    })

    # Compute per-class IoU
    num_labels = cm_test.shape[0]
    per_class_iou = []
    for k in range(1, num_labels):
        TP = cm_test[k, k]
        FP = cm_test[:, k].sum() - TP
        FN = cm_test[k, :].sum() - TP
        denom = TP + FP + FN
        iou_k = TP / denom if denom > 0 else float("nan")
        per_class_iou.append(iou_k)

    # Compute mean IoU over all non‐background classes
    valid_iou = [v for v in per_class_iou if not np.isnan(v)]
    mean_iou = np.mean(valid_iou)
    wandb.log({
        "mean_IoU": mean_iou,
        **{f"IoU/class_{k}": per_class_iou[k-1] for k in range(1, num_labels)}
    })

    # Save final checkpoint
    os.makedirs(args.output_dir, exist_ok=True)

    pretrained_short = os.path.basename(args.pretrained).replace("/", "-")
    filename = (
        f"{pretrained_short}_"
        f"{args.ign_background}_"
        f"{now_str}_"
        f"nl{args.num_labels}_"
        f"e{args.epochs}_"
        f"bs{args.batch_size}_"
        f"lr{args.lr:.0e}_"
        f"is{args.img_size}.bin"
    )

    save_path = os.path.join(args.output_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")
    wandb.save(save_path)
    wandb.finish()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kaggle_base = os.path.normpath(os.path.join(base_dir, "..", "datasets", "kaggle-image-segmentation"))

    p = argparse.ArgumentParser(description="SegFormer finetuning with train/valid/test")

    # hardcoded dataset dirs
    p.set_defaults(train_images = os.path.join(kaggle_base, "train", "images"))
    p.set_defaults(train_masks  = os.path.join(kaggle_base, "train", "masks"))
    p.set_defaults(valid_images = os.path.join(kaggle_base, "valid",  "images"))
    p.set_defaults(valid_masks  = os.path.join(kaggle_base, "valid",  "masks"))
    p.set_defaults(test_images  = os.path.join(kaggle_base, "test",  "images"))
    p.set_defaults(test_masks   = os.path.join(kaggle_base, "test",  "masks"))

    p.add_argument("--train_images",  type=str, help="Folder w/ train RGB images")
    p.add_argument("--train_masks",   type=str, help="Folder w/ train masks (L-mode)")
    p.add_argument("--valid_images",  type=str, help="Folder w/ valid RGB images")
    p.add_argument("--valid_masks",   type=str, help="Folder w/ valid masks")
    p.add_argument("--test_images",   type=str, help="Folder w/ test RGB images")
    p.add_argument("--test_masks",    type=str, help="Folder w/ test masks")

    # default arguments
    p.add_argument("--pretrained",    type=str, default="nvidia/segformer-b4-finetuned-ade-512-512", help="Pretrained SegFormer checkpoint")
    p.add_argument("--num_labels",    type=int, default=53, help="Number of classes")
    p.add_argument("--img_size",      type=int, default=512, help="Resize H and W to this")
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--wd",            type=float, default=0.01, help="Weight decay for AdamW (default: 0.01)")
    p.add_argument("--epochs",        type=int, default=3)
    p.add_argument("--ign_background", action="store_true", default=False, help="Ignore background pixels (default: False)")
    p.add_argument("--wandb_project", type=str, default="cs231n_eye_in_the_sky-segformer", help="W&B project name")

    # required args
    p.add_argument("--output_dir",    type=str, required=True, help="Where to save finetuned model")
    
    args = p.parse_args()
    train_validate_test(args)
