#!/usr/bin/env python3

import os
import torch
import argparse
import datetime
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.optim import AdamW
class SegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, feature_extractor, size=512):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.extractor  = feature_extractor
        self.size       = size

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

        # Use the feature extractor to: resize to self.size, normalize, convert to tensor.
        encoding = self.extractor(
            images=img,
            size=self.size,            # both height and width → self.size (e.g. 512)
            return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze(0)  # shape: (3, self.size, self.size)

        # Resize the mask the same way, with nearest neighbor (to preserve integer labels).
        msk_resized = msk.resize((self.size, self.size), resample=Image.NEAREST)
        msk_resized_np = np.array(msk_resized, dtype=np.int64)
        labels = torch.from_numpy(msk_resized_np).long()    # shape: (self.size, self.size)

        return pixel_values, labels

def train_validate_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load feature extractor + model
    extractor = SegformerImageProcessor.from_pretrained(args.pretrained)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.pretrained,
        num_labels=args.num_labels,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    model.config.ignore_index = 0  # Set ignore_index to 0 (background) for loss calculation

    # Prepare Datasets and DataLoaders
    train_ds = SegDataset(
        images_dir=args.train_images,
        masks_dir=args.train_masks,
        feature_extractor=extractor,
        size=args.img_size,
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

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_accum = 0.0
        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(device)  # (B, 3, H, W)
            labels       = labels.to(device)        # (B, H, W)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} — Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        valid_loss_accum = 0.0
        with torch.no_grad():
            for pixel_values, labels in valid_loader:
                pixel_values = pixel_values.to(device)
                labels       = labels.to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                valid_loss_accum += outputs.loss.item()

        avg_valid_loss = valid_loss_accum / len(valid_loader)
        print(f"Epoch {epoch}/{args.epochs} — Valid Loss: {avg_valid_loss:.4f}")

    # Final Evaluation
    model.eval()
    test_loss_accum = 0.0
    with torch.no_grad():
        for pixel_values, labels in test_loader:
            pixel_values = pixel_values.to(device)
            labels       = labels.to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            test_loss_accum += outputs.loss.item()

    avg_test_loss = test_loss_accum / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Save final checkpoint
    os.makedirs(args.output_dir, exist_ok=True)

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pretrained_short = os.path.basename(args.pretrained).replace("/", "-")
    filename = (
        f"{pretrained_short}_"
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
    p.add_argument("--pretrained",    type=str, default="nvidia/segformer-b4-finetuned-ade-512-512",
                   help="Pretrained SegFormer checkpoint")
    p.add_argument("--num_labels",    type=int, default=53, help="Number of classes")
    p.add_argument("--img_size",      type=int, default=512, help="Resize H and W to this")
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--epochs",        type=int, default=3)

    # required args
    p.add_argument("--output_dir",    type=str, required=True, help="Where to save finetuned model")
    
    args = p.parse_args()

    train_validate_test(args)
