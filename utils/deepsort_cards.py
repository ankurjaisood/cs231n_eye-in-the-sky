import os
import cv2
import time
import argparse
import torch
import warnings
import json
import sys
import yaml
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/fast-reid'))

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results


def load_classes_from_yaml(yaml_path):
    """Load class names and count from YAML file"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if 'names' not in data:
            raise ValueError(f"'names' key not found in {yaml_path}")
        if 'nc' not in data:
            raise ValueError(f"'nc' key not found in {yaml_path}")

        class_names = data['names']
        num_classes = data['nc']

        # Validate consistency
        if len(class_names) != num_classes:
            print(f"Warning: Number of class names ({len(class_names)}) doesn't match nc ({num_classes})")
            print(f"Using actual number of class names: {len(class_names)}")
            num_classes = len(class_names)

        return class_names, num_classes

    except Exception as e:
        print(f"Error loading classes from {yaml_path}: {e}")
        # Fallback to default 52 card classes
        print("Falling back to default 52 card classes")
        default_classes = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s',
                          '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
                          '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s',
                          'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks',
                          'Qc', 'Qd', 'Qh', 'Qs']
        return default_classes, 52


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        # Load card classes from YAML file
        self.card_classes, self.num_classes = load_classes_from_yaml(args.data_yaml)
        self.idx_to_class = {str(i): self.card_classes[i] for i in range(len(self.card_classes))}

        print(f"Loaded {self.num_classes} classes from {args.data_yaml}: {self.card_classes}")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda, segment=self.args.segment)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

        # Initialize frame counter for tracking memory management (optional feature)
        self.frames_without_detections = 0
        self.max_frames_before_reset = args.max_frames_before_reset
        self.memory_clearing_enabled = args.max_frames_before_reset is not None and args.max_frames_before_reset > 0

        # Initialize class counting for accuracy evaluation
        self.tracked_instances = {}  # track_id -> class_name mapping to avoid double counting
        self.class_counts = {}  # final counts per class

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            # TODO save masks

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

        # Close video writer
        if hasattr(self, 'writer') and self.writer:
            self.writer.release()

        # Re-encode to x264 mp4 if save_path was used
        if self.args.save_path and hasattr(self, 'save_video_path'):
            self._reencode_to_x264_mp4()

    def _reencode_to_x264_mp4(self):
        """Re-encode the MJPG AVI video to x264 MP4 format"""
        if not os.path.exists(self.save_video_path):
            self.logger.warning(f"Original video file not found: {self.save_video_path}")
            return

        # Define output MP4 path
        mp4_path = os.path.join(self.args.save_path, "results.mp4")

        # Prioritize system ffmpeg over conda env version
        if os.path.exists('/usr/bin/ffmpeg'):
            ffmpeg_cmd = '/usr/bin/ffmpeg'
            self.logger.info("Using system ffmpeg at /usr/bin/ffmpeg")
        else:
            ffmpeg_cmd = 'ffmpeg'
            self.logger.info("Using ffmpeg from PATH")

        try:
            self.logger.info("Re-encoding video to x264 MP4...")

            # FFmpeg command for x264 encoding with good quality settings
            cmd = [
                ffmpeg_cmd, '-y',  # -y to overwrite output file
                '-i', self.save_video_path,  # input file
                '-c:v', 'libx264',  # x264 codec
                '-preset', 'medium',  # encoding speed/compression tradeoff
                '-crf', '23',  # quality setting (lower = better quality)
                '-pix_fmt', 'yuv420p',  # pixel format for compatibility
                mp4_path  # output file
            ]

            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if os.path.exists(mp4_path):
                # Get file sizes for comparison
                avi_size = os.path.getsize(self.save_video_path) / (1024 * 1024)  # MB
                mp4_size = os.path.getsize(mp4_path) / (1024 * 1024)  # MB

                self.logger.info(f"Re-encoding successful!")
                self.logger.info(f"Original AVI: {avi_size:.1f} MB")
                self.logger.info(f"X264 MP4: {mp4_size:.1f} MB")
                self.logger.info(f"Saved to: {mp4_path}")

                # Optionally remove the original AVI file
                try:
                    os.remove(self.save_video_path)
                    self.logger.info("Original AVI file removed")
                except Exception as e:
                    self.logger.warning(f"Could not remove original AVI: {e}")
            else:
                self.logger.error("Re-encoding failed - output file not created")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg failed with return code {e.returncode}")
            self.logger.error(f"Error output: {e.stderr}")
        except FileNotFoundError:
            self.logger.error("FFmpeg not found. Please install FFmpeg to enable x264 re-encoding.")
            self.logger.info("The original AVI file is still available at: {}".format(self.save_video_path))
        except Exception as e:
            self.logger.error(f"Unexpected error during re-encoding: {e}")

    def run(self):
        results = []
        idx_frame = 0

        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            if self.args.segment:
                bbox_xywh, cls_conf, cls_ids, seg_masks = self.detector(im)
            else:
                bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # Filter detections based on loaded classes
            if len(bbox_xywh) > 0:
                # Optional: filter by confidence threshold
                conf_threshold = getattr(self.args, 'conf_threshold', 0.5)
                mask = cls_conf >= conf_threshold

                # Optional: filter by specific card classes if needed
                if hasattr(self.args, 'target_classes') and self.args.target_classes:
                    class_mask = torch.zeros_like(cls_ids, dtype=torch.bool)
                    for target_class in self.args.target_classes:
                        class_mask |= (cls_ids == target_class)
                    mask = mask & class_mask
                else:
                    # Accept all valid classes based on loaded YAML
                    valid_classes = (cls_ids >= 0) & (cls_ids < self.num_classes)
                    mask = mask & valid_classes

                bbox_xywh = bbox_xywh[mask]
                cls_conf = cls_conf[mask]
                cls_ids = cls_ids[mask]

                # Remove bbox dilation for cards (was specific to pedestrian detection)
                # bbox_xywh[:, 2:] *= 1.2  # REMOVED

                if self.args.segment and len(bbox_xywh) > 0:
                    seg_masks = seg_masks[mask]

            # Handle tracker memory management based on detections (only if enabled)
            if self.memory_clearing_enabled:
                if len(bbox_xywh) > 0:
                    # Reset counter when detections are present
                    self.frames_without_detections = 0
                else:
                    # Increment counter when no detections
                    self.frames_without_detections += 1

                    # Clear tracker memory if no detections for specified frames
                    if self.frames_without_detections >= self.max_frames_before_reset:
                        self._clear_tracker_memory()
                        self.frames_without_detections = 0  # Reset counter after clearing
                        self.logger.info(f"Cleared tracker memory after {self.max_frames_before_reset} frames without detections")

            # do tracking
            if len(bbox_xywh) > 0:
                if self.args.segment:
                    outputs, mask_outputs = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im, seg_masks)
                else:
                    outputs, _ = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)
            else:
                outputs = []
                mask_outputs = None

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                cls = outputs[:, -2]
                names = [self.idx_to_class.get(str(int(label)), f"class_{int(label)}") for label in cls]

                # Track unique instances of each class for counting
                for identity, class_id in zip(identities, cls):
                    track_id = int(identity)
                    class_name = self.idx_to_class.get(str(int(class_id)), f"class_{int(class_id)}")

                    # Only count each track_id once per class
                    if track_id not in self.tracked_instances:
                        self.tracked_instances[track_id] = class_name
                        if class_name not in self.class_counts:
                            self.class_counts[class_name] = 0
                        self.class_counts[class_name] += 1

                ori_im = draw_boxes(ori_im, bbox_xyxy, names, identities, None if not self.args.segment else mask_outputs)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities, cls))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            if len(results) > 0:
                write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), len(bbox_xywh) if len(bbox_xywh) > 0 else 0, len(outputs)))

        # Output final class counts
        self._output_class_counts()

    def _clear_tracker_memory(self):
        """Clear all tracker memory including tracks and reset ID counter"""
        # Clear all active tracks
        self.deepsort.tracker.tracks = []

        # Reset the ID counter to start fresh
        self.deepsort.tracker._next_id = 1

        # Clear the distance metric memory if it has samples
        if hasattr(self.deepsort.tracker.metric, 'samples'):
            self.deepsort.tracker.metric.samples = {}

    def _output_class_counts(self):
        """Output the final counts of each class detected in the video"""
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)

        if not self.class_counts:
            print("No classes detected in this video.")
            return

        # Sort by class name for consistent output
        sorted_classes = sorted(self.class_counts.items())

        print("Total unique instances detected:")
        for class_name, count in sorted_classes:
            print(f"  {class_name}: {count}")

        print(f"\nTotal unique tracks: {len(self.tracked_instances)}")

        # Save to file if save_path is specified
        if self.args.save_path:
            counts_file = os.path.join(self.args.save_path, "class_counts.json")
            try:
                with open(counts_file, 'w') as f:
                    json.dump(self.class_counts, f, indent=2)
                print(f"\nClass counts saved to: {counts_file}")
            except Exception as e:
                self.logger.warning(f"Could not save class counts: {e}")

        print("="*50)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default='demo.avi')
    parser.add_argument("--config_mmdetection", type=str, default="./configs/mmdet.yaml")
    parser.add_argument("--config_detection", type=str, default="./configs/mask_rcnn.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str, default="./configs/fastreid.yaml")
    parser.add_argument("--fastreid", action="store_true")
    parser.add_argument("--mmdet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")

    # Card-specific arguments
    parser.add_argument("--data_yaml", type=str, required=True,
                       help="Path to data YAML file containing class names (e.g., data_hilo.yaml or data.yaml)")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--target_classes", nargs="+", type=int, help="Specific card class IDs to track (default: all)")
    parser.add_argument("--max_frames_before_reset", type=int, default=None,
                       help="Number of consecutive frames without detections before clearing tracker memory (default: disabled)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    if args.segment:
        cfg.USE_SEGMENT = True
    else:
        cfg.USE_SEGMENT = False
    if args.mmdet:
        cfg.merge_from_file(args.config_mmdetection)
        cfg.USE_MMDET = True
    else:
        cfg.merge_from_file(args.config_detection)
        cfg.USE_MMDET = False
    cfg.merge_from_file(args.config_deepsort)
    if args.fastreid:
        cfg.merge_from_file(args.config_fastreid)
        cfg.USE_FASTREID = True
    else:
        cfg.USE_FASTREID = False

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()