import torch
import wandb
import os
from ultralytics import YOLO

W_AND_B_API_KEY = os.getenv("W_AND_B_API_KEY")
MODEL_PATH = './models/yolov10m.pt'
DATASET_PATH = './datasets/kaggle-image-detection/data.yaml'

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

if W_AND_B_API_KEY is None:
    raise ValueError("W_AND_B_API_KEY not set in environment variables.")

wandb_api = wandb.login(key=W_AND_B_API_KEY)
print("Weights & Biases:", wandb_api)
if wandb_api is None:
    raise ValueError("Failed to authenticate with Weights & Biases.")

epochs = [10, 25, 50]
weight_decay = [1e-4, 1e-5, 1e-6]
learning_rate = [1e-2, 1e-3, 1e-4]

for epoch, wd, lr in zip(epochs, weight_decay, learning_rate):
    model = YOLO(MODEL_PATH)

    train_results = model.train(
        data=DATASET_PATH,  # Path to dataset configuration file
        epochs=epoch,  # Number of training epochs
        lr0=lr,  # Initial learning rate
        weight_decay=wd,  # Weight decay
        imgsz=416,  # Image size for training
        device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        project="cs231n_eye_in_the_sky", 
        name="milestone_tests"
    )

    # Perform object detection on an image
    results = model("./datasets/kaggle-image-detection/test/images/000246247_jpg.rf.fb915aef7c063ce2ac971f8de0d8b2c1.jpg")  # Predict on an image
    results[0].show()  # Display results

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model
