from ultralytics import YOLO

# Path to dataset yaml file
DATA_YAML = "data.yaml"

# Load YOLOv8 pretrained model (small model recommended for start)
model = YOLO("yolov8s.pt")

# Train the model
model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    device="cpu"   # <--- add this
)


