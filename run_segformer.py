import os
import torch
import cv2
import numpy as np
from transformers import (
    SegformerForSemanticSegmentation, 
    SegformerImageProcessor, 
    SegformerConfig
)
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort

BIN_PATH = None

# original model for 10 epochs
BIN_PATH = "/home/anksood/cs231n/cs231n_eye-in-the-sky/models/segformer_checkpoints/lr5e-5_bs8_ep10_20250530_224702/segformer-b4-finetuned-ade-512-512_20250530_231126_nl53_e10_bs8_lr5e-05_is512.bin"

# better model for 25 epochs
BIN_PATH = "/home/anksood/cs231n/cs231n_eye-in-the-sky/models/segformer_checkpoints/lr5e-5_bs8_ep25_20250531_194451/segformer-b4-finetuned-ade-512-512_20250531_194456_nl53_e25_bs8_lr5e-05_is512.bin"

# ignore background pixels for 10 epochs (didnt work well)
#BIN_PATH = "/home/anksood/cs231n/cs231n_eye-in-the-sky/models/segformer_checkpoints/lr5e-5_bs8_ep10_20250531_135500/segformer-b4-finetuned-ade-512-512_20250531_142024_nl53_e10_bs8_lr5e-05_is512.bin"

# ignore background pixels for 20 epochs (worked ok)
#BIN_PATH = "/home/anksood/cs231n/cs231n_eye-in-the-sky/models/segformer_checkpoints/lr5e-5_bs8_ep20_20250531_142026/segformer-b4-finetuned-ade-512-512_20250531_151105_nl53_e20_bs8_lr5e-05_is512.bin"

MODEL_NAME = "nvidia/segformer-b4-finetuned-ade-512-512"

#VIDEO_IN  = "/home/anksood/cs231n/cs231n_eye-in-the-sky/git_datasets/clips/2_416px_30fps.mp4"
VIDEO_IN  = "/home/anksood/cs231n/cs231n_eye-in-the-sky/git_datasets/11_416px_10fps.mp4"
VIDEO_OUT = "./output5.mp4"
IMG_SIZE = 512
BATCH_SIZE = 32

MIN_BB_AREA = 250
MIN_PIXEL_REGION_AREA = 1500  # minimum area of a pixel region to be considered valid
BB_CONFIDENCE_THRESHOLD = 0.25  # minimum confidence score for bounding boxes

NUM_CLASSES = 53

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
    
def run_segformer(images):
    feature_extractor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

    inputs = feature_extractor(images=images, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    return logits, predictions

def run_segformer_from_bin(images):
    # Load the base config from the Hub
    config = SegformerConfig.from_pretrained(MODEL_NAME)  
    config.num_labels = 53
    config.ignore_mismatched_sizes = True

    # Instantiate an empty SegFormerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation(config)

    # Load local state_dict into that model
    state_dict = torch.load(BIN_PATH, map_location="cpu")
    model.load_state_dict(state_dict)  # must match exactly the layer-names in `config`
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    extractor = SegformerImageProcessor.from_pretrained(MODEL_NAME)

    # Forward pass
    inputs = extractor(images=images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    logits      = outputs.logits
    predictions = torch.argmax(logits, dim=1)  # (B, H_out, W_out)
    return logits, predictions

def extract_frames_from_video(video_path: str) -> (list, float):
    """
    Read all frames from `video_path` and return a list of BGR numpy arrays,
    plus the video FPS. 
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames.append(frame_bgr.copy())
    cap.release()
    print(f"Extracted {len(frames)} frames at {fps} FPS from {video_path}")
    return frames, fps

def frames_to_resized_pil(frames: list, size: int) -> list:
    """
    Convert a list of BGR numpy frames to a list of PIL Images (RGB), each resized
    to (size, size). Returns a list of PIL.Image.
    """
    pil_list = []
    for frame_bgr in frames:
        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_resized = pil_img.resize((size, size), resample=Image.BILINEAR)
        pil_list.append(pil_resized)
    return pil_list

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a 2D array of class indices (H×W, int) into a BGR color image
    using OpenCV's JET colormap. Returns a (H, W, 3) uint8 image.
    """
    mask_flat = mask.astype(np.float32).flatten()
    min_id, max_id = mask_flat.min(), mask_flat.max()
    if max_id == min_id:
        scaled = np.zeros_like(mask, dtype=np.uint8)
    else:
        scaled = ((mask - min_id) / (max_id - min_id) * 255.0).astype(np.uint8)

    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)  # BGR
    return colored

def create_output_video_from_masks(
    original_frames: list,
    logits: torch.Tensor,
    masks: torch.Tensor,
    output_path: str,
    fps: float
):
    """
    Given a list of original BGR frames (H_orig × W_orig × 3), and a torch.Tensor `masks`
    of shape (N, H_mask, W_mask) containing integer class indices,
    up‐sample each mask to the original frame size, colorize it, and write side‐by‐side
    to `output_path` at the given `fps`.
    """
    num_frames = len(original_frames)
    if masks.shape[0] != num_frames:
        raise ValueError(
            f"Number of masks ({masks.shape[0]}) does not match number of frames ({num_frames})"
        )

    # Determine original frame size
    orig_h, orig_w = original_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    final_w = orig_w * 2
    final_h = orig_h

    writer = cv2.VideoWriter(output_path, fourcc, fps, (final_w, final_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter at {output_path}")

    # SORT tracker initalization
    max_age_num_frames = int(fps * 0.5)
    num_frames_new_track = max(1, int(fps) // 2)
    print(f"Initializing DeepSORT tracker with max_age={max_age_num_frames}, n_init={num_frames_new_track}")
    tracker = DeepSort(
        max_age=max_age_num_frames,
        n_init=num_frames_new_track, 
        max_cosine_distance=0.2,
        embedder="clip_RN50x4",
        embedder_gpu=torch.cuda.is_available()
    )

    for idx in range(num_frames):
        frame_bgr = original_frames[idx]
        logits_small = logits[idx]
        mask_small = masks[idx].cpu().numpy().astype(np.int32)  # (IMG_SIZE, IMG_SIZE)

        # Compute probs over mask
        with torch.no_grad():
            probs_small = torch.softmax(logits_small, dim=0).cpu().numpy()

        # Upsample mask back to original size
        mask_full = cv2.resize(
            mask_small.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        # Colorize the upsampled mask
        color_mask = colorize_mask(mask_full)

        frame_detections = []
        frame_masks = []
        # For each unique class in mask_full, find contours and draw boxes, labels
        unique_ids = np.unique(mask_full)
        for class_id in unique_ids:
            # Skip background
            if class_id == 0:
                continue

            # Upsample the probability map for this class to full resolution
            prob_small = probs_small[class_id]  # shape: (H_mask, W_mask)
            prob_full = cv2.resize(
                prob_small.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR
            )  # shape: (orig_h, orig_w)

            class_name = None
            if NUM_CLASSES == 53:
                class_name = id2label[class_id]
            elif NUM_CLASSES == 13:
                class_name = id2label_suits_category[class_id]
            elif NUM_CLASSES == 4:
                class_name = id2label_category[class_id]
            else:
                raise ValueError(f"Unsupported number of classes: {NUM_CLASSES}")
            
            # Create a binary mask for this class
            binary = (mask_full == class_id).astype(np.uint8) * 255
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

            # Find contours (connected components); OpenCV expects 8‐bit single channel
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            filtered_contours = []
            for cnt in contours:
                area_cc = cv2.contourArea(cnt)
                if area_cc >= MIN_PIXEL_REGION_AREA:
                    filtered_contours.append(cnt)
                else:
                    print(f"Skipping small contour with area {area_cc} < {MIN_PIXEL_REGION_AREA}")

            # Draw a box + the class name for each contour
            for cnt in filtered_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area < MIN_BB_AREA:
                    print(f"Skipping small box with area {area} < {MIN_BB_AREA}")
                    continue

                # Build a mask for this contour to extract pixel probabilities
                mask_roi = np.zeros((orig_h, orig_w), dtype=np.uint8)
                cv2.drawContours(mask_roi, [cnt], -1, 1, thickness=-1)  # fill region
                pixel_probs = prob_full[mask_roi == 1]
                score = float(pixel_probs.mean())

                if score < BB_CONFIDENCE_THRESHOLD:
                    print(f"Skipping box with low score {score:.2f} < {BB_CONFIDENCE_THRESHOLD}")
                    continue

                x1, y1, x2, y2 = x, y, x + w, y + h
                frame_detections.append(([x, y, w, h], score, class_name))
                frame_masks.append(mask_roi)

                # Draw rectangle on color_mask (right side) or on blended image
                cv2.rectangle(
                    color_mask,
                    (x1, y1),
                    (x2, y2),
                    color=(0, 255, 0),
                    thickness=2
                )
                # Put class name and confidence text slightly above the top‐left corner of the box
                text = f"{class_name}:{score:.2f}"
                text_pos = (x, y - 5 if y - 5 > 5 else y + 15)
                cv2.putText(
                    color_mask,
                    text,
                    text_pos,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
        
        # Run DeepSORT tracker on the detected bounding boxes
        print(f"Frame {idx+1}/{num_frames}: Found {len(frame_detections)} detections")
        # Run tracker
        assert len(frame_detections) == len(frame_masks)
        tracks = tracker.update_tracks(frame_detections, frame=frame_bgr, instance_masks=frame_masks)

        # Draw the tracked bounding boxes on the original frame
        for track in tracks:
            if not track.is_confirmed():
                print(f"Skipping unconfirmed track {track.track_id}")
                continue
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            tid = track.track_id

            cv2.rectangle(
                frame_bgr,
                (x1, y1),
                (x2, y2),
                color=(0, 0, 255),
                thickness=2
            )
            cv2.putText(
                frame_bgr,
                f"ID:{tid}",
                (x1, y1 - 5 if y1 - 5 > 5 else y1 + 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(255, 255, 255),
                thickness=1
            )


        # Create side by side composite
        combined = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        combined[:, :orig_w, :] = frame_bgr
        combined[:, orig_w:, :] = color_mask

        writer.write(combined)

        '''
        # Optional: display in real-time
        cv2.imshow("Original | Segmentation", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        '''

        if (idx + 1) % 50 == 0:
            print(f"Written {idx + 1}/{num_frames} frames to video...")

    writer.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to: {output_path}")

def process_video(video_in_path: str, video_out_path: str):
    """
    1. Extract all frames from input video into a list.
    2. Convert those frames to resized PIL images (IMG_SIZE x IMG_SIZE).
    3. Run SegFormer inference in batches over the list of PIL images.
    4. Postprocess: upsample & colorize masks, create output video.
    """
    # 1) Extract frames
    original_frames, fps = extract_frames_from_video(video_in_path)
    if len(original_frames) == 0:
        raise RuntimeError(f"No frames extracted from {video_in_path}")
    pil_images = frames_to_resized_pil(original_frames, IMG_SIZE)

    # 3) Run inference in batches
    all_masks = []
    all_logits = []
    num_frames = len(pil_images)
    for start_idx in range(0, num_frames, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_frames)
        batch = pil_images[start_idx:end_idx]
        if BIN_PATH and os.path.isfile(BIN_PATH):
            print(f"Running batch {start_idx}..{end_idx-1} through local .bin model...")
            logits_batch, masks_batch = run_segformer_from_bin(batch)  # (batch_size, IMG_SIZE, IMG_SIZE)
        else:
            print(f"Running batch {start_idx}..{end_idx-1} through Hub model...")
            logits_batch, masks_batch = run_segformer(batch)  # (batch_size, IMG_SIZE, IMG_SIZE)
        all_logits.append(logits_batch)
        all_masks.append(masks_batch)
        torch.cuda.empty_cache()

    logits = torch.cat(all_logits, dim=0)
    masks = torch.cat(all_masks, dim=0)
    create_output_video_from_masks(original_frames, logits, masks, video_out_path, fps)

if __name__ == "__main__":
    if not os.path.isfile(VIDEO_IN):
        raise FileNotFoundError(f"Input video file does not exist: {VIDEO_IN}")
    process_video(VIDEO_IN, VIDEO_OUT)
