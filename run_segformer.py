import torch
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from PIL import Image

def run_segformer(images):
    model_name = "nvidia/segformer-b4-finetuned-ade-512-512"
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)

    inputs = feature_extractor(images=images, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    return predictions

def visualize_segmentation_masks(images, masks):
    for img, mask in zip(images, masks):
        mask_np = mask.cpu().numpy()

        plt.figure(figsize=(12,5))
        # original
        plt.subplot(1,2,1)
        plt.title("Input")
        plt.imshow(img)
        plt.axis("off")

        # segmentation
        plt.subplot(1,2,2)
        plt.title("Segmentation")
        plt.imshow(mask_np, cmap="nipy_spectral", interpolation="nearest")
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    image_paths = ["./datasets/kaggle-image-detection/test/images/000246247_jpg.rf.fb915aef7c063ce2ac971f8de0d8b2c1.jpg",
                   "./datasets/kaggle-image-segmentation/images/00000.png",
                   "./datasets/kaggle-image-segmentation/images/00018.png"]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    resized_images = [img.resize((512, 512), resample=Image.BILINEAR) for img in images]

    # Run Segmformer
    masks = run_segformer(resized_images)
    
    # Visualize the segmentation masks
    visualize_segmentation_masks(resized_images, masks)

