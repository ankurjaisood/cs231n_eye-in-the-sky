import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

MODEL_PATH = './models/yolov10m.pt'
DATASET_PATH = './datasets/kaggle-image-detection/data.yaml'

from ultralytics import YOLO

model = YOLO(MODEL_PATH)

train_results = model.train(
    data=DATASET_PATH,  # Path to dataset configuration file
    epochs=10,  # Number of training epochs
    imgsz=416,  # Image size for training
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Perform object detection on an image
results = model("./datasets/kaggle-image-detection/test/images/000246247_jpg.rf.fb915aef7c063ce2ac971f8de0d8b2c1.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model
