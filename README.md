# Demo Video
https://www.youtube.com/watch?v=yUcEbM6uDno

# Report
https://cs231n.stanford.edu/papers/text_file_840589501-CS231n_Final_Project.pdf

# How to run:
1. Gather datasets at ./datasets (see README in ./datasets)
2. (For yolov10) Clone Ultralytics repo: git@github.com:ultralytics/ultralytics.git @ ./
3. (For wandb) Export W_AND_B_API_KEY="YOUR_API_KEY"
4. yolo settings wandb=True

# How to get datasets from git_datasets:
```
git submodule update --init
cd git_datasets
git pull origin master
```

# Finetune YOLOv10 (milestone):
5. python run_yolo.py

# Finetune SegFormer:
5. ./segformer.sh 
OR
6. ./segformer_4.sh (4 model version)
OR
7. ./segformer_13.sh (13 model version)

# Finetune YOLOv5:
8. ./utils/yolo.py

# Run Segformer:
9. ./run_segformer.py

# Run YOLOv5:
9. ./run_all_clips.sh (required cloning yolov5 repo)

# Run YOLOv5l:
9. ./scrips/run_model.sh (required cloning yolov5 repo)
