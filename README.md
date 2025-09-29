# 🚗 License Plate Recognition (YOLOv8 + EasyOCR)

## 📌 Overview
This project implements an **Automatic License Plate Recognition (ALPR)** system.  
It uses **YOLOv8** for license plate detection and **EasyOCR** for text recognition.  

### What it does:
- Detects plates in images and videos  
- Reads the plate numbers using OCR  
- Saves results as annotated images, cropped plates, and structured CSV logs  

### Tech Stack 
- OpenCV  
- Pandas  

---

## 📂 Project Structure
├── lpr_image.py # Main detection + OCR script
├── lpr_utils.py # Helper functions
├── train.py # YOLO training script
├── assets/
│ ├── inputs/ # Sample images/videos
│ └── outputs/ # Results (vis, crops, preds.csv)
├── requirements.txt # Dependencies
└── README.md
##2. Create & activate conda environment
conda create -n lpr python=3.9 -y
conda activate lpr
## 3. Install dependencies
pip install -r requirements.txt
## 4 Run on a single image
python lpr_image.py --source assets/inputs/car.jpg --weights runs/detect/train8/weights/best.pt --device cpu
## 5 Run on a video
python lpr_image.py --source assets/inputs/car.mp4 --weights runs/detect/train8/weights/best.pt --device cpu
#  Outputs

Annotated frames/video → assets/outputs/vis/

Cropped license plates → assets/outputs/crops/

Detection + OCR CSV → assets/outputs/preds.csv
#Roadmap

 Image support

 Video support

 CSV logging

 Live webcam support (--source 0)

 Streamlit deployment for interactive UI
#License

For educational and research purposes

Acknowledge Ultralytics YOLOv8
 and EasyOCR 
 
