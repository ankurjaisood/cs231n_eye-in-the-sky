#!/bin/bash
#python track.py --source $1 --yolo_model ~/yolov5/runs/train/exp4/weights/best.pt --img $2 --deep_sort_model resnet50_MSMT17 --show-vid --save-vid
sz=`ffprobe -v quiet -show_format -show_streams $1 | grep -w width | cut -f2 -d=`
echo $sz
#python track.py --source $1 --yolo_model ~/yolov5/runs/train/exp4/weights/best.pt --img $sz --deep_sort_model osnet_x0_5_market1501 --show-vid --save-vid
rm -rf /home/ubuntu/output/*
#python track.py --source $1 --yolo_model ~/yolov5/runs/train/exp4/weights/best.pt --img $((sz * 2)) --deep_sort_model mobilenetv2_x1_4 --conf-thres 0.70 --iou-thres 0.45 --show-vid --save-vid
#python track.py --source $1 --yolo_model ~/yolov5/runs/train/exp4/weights/best.pt --img $((sz * 2)) --deep_sort_model resnet50 --conf-thres 0.70 --iou-thres 0.45 --save-vid
#python track.py --source $1 --yolo_model ~/yolov5/runs/train/exp11/weights/best.pt --img $((sz * 2)) --deep_sort_model resnet50 --conf-thres 0.88 --save-vid #--show-vid
python track.py --source $1 --yolo_model ~/yolov5/runs/train/exp11/weights/best.pt --img $((sz * 2)) --deep_sort_model resnet50 --conf-thres 0.9 --iou-thres 0.2 --save-vid #--show-vid
#python track.py --source $1 --yolo_model ~/yolov5/runs/train/exp11/weights/best.pt --img $((sz * 2)) --deep_sort_model resnet50 --conf-thres 0.7 --save-vid --show-vid
#python track.py --source $1 --yolo_model ~/yolov5/runs/train/exp4/weights/best.pt --img $((sz * 2)) --deep_sort_model resnet50_fc512 --conf-thres 0.65 --show-vid --save-vid
#python track.py --source $1 --yolo_model ~/yolov5/runs/train/exp4/weights/best.pt --img $((sz * 2)) --deep_sort_model resnet50 --show-vid --save-vid
