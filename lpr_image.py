# lpr_image.py
from pathlib import Path
import argparse
import csv
import cv2
import torch
from ultralytics import YOLO
import easyocr
import os

from lpr_utils import ensure_dir, pad_and_clip_box, preprocess_for_ocr, normalize_plate_text, pick_best_candidate


def is_image(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_sources(source: Path):
    """Return a list of image paths, or a video flag + path"""
    abs_path = str(source)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Source not found: {abs_path}")

    # video
    if source.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        return "video", abs_path

    # single image
    if is_image(source):
        return "images", [source]

    # folder of images
    if source.is_dir():
        imgs = [p for p in source.iterdir() if is_image(p)]
        return "images", imgs

    raise ValueError(f"Unsupported source type: {abs_path}")


def process_images(images, model, reader, args, vis_dir, crops_dir, csv_path, device):
    first_write = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if first_write:
            writer.writerow(["source", "x1", "y1", "x2", "y2",
                             "det_conf", "ocr_text", "ocr_conf", "crop_path", "vis_path"])

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Cannot read image: {img_path}")
                continue
            H, W = img.shape[:2]

            results = model.predict(source=str(img_path),
                                    conf=args.conf, iou=args.iou,
                                    device=device, verbose=False)
            r = results[0]

            annotated = r.plot()
            vis_out = vis_dir / Path(img_path).name
            ensure_dir(vis_out)
            cv2.imwrite(str(vis_out), annotated)

            if not r.boxes or len(r.boxes) == 0:
                writer.writerow([str(img_path), "", "", "", "", "", "", "", "", str(vis_out)])
                print(f"[INFO] No detections: {Path(img_path).name}")
                continue

            for i, (xyxy, conf) in enumerate(zip(r.boxes.xyxy.cpu().numpy(),
                                                 r.boxes.conf.cpu().numpy())):
                x1, y1, x2, y2 = map(int, xyxy)
                x1p, y1p, x2p, y2p = pad_and_clip_box(x1, y1, x2, y2,
                                                      args.pad, W, H)
                crop = img[y1p:y2p, x1p:x2p]

                if crop.size == 0:
                    writer.writerow([str(img_path), x1, y1, x2, y2,
                                     float(conf), "", 0.0, "", str(vis_out)])
                    continue

                crop_for_ocr = crop if args.no_preproc else preprocess_for_ocr(crop)
                ocr_results = reader.readtext(crop_for_ocr)
                candidates = [(normalize_plate_text(text), float(c))
                              for _, text, c in ocr_results if normalize_plate_text(text)]
                best_text, best_conf = pick_best_candidate(candidates)

                if best_conf < args.ocr_min_conf:
                    best_text = ""

                crop_name = f"{Path(img_path).stem}_plate{i}.jpg"
                crop_out = crops_dir / crop_name
                ensure_dir(crop_out)
                cv2.imwrite(str(crop_out), crop)

                writer.writerow([str(img_path), x1p, y1p, x2p, y2p,
                                 float(conf), best_text, best_conf,
                                 str(crop_out), str(vis_out)])

            print(f"[OK] {Path(img_path).name} ➜ {vis_out.name}")


def process_video(video_path, model, reader, args, vis_dir, crops_dir, csv_path, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    # prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = vis_dir / (Path(video_path).stem + "_out.mp4")
    ensure_dir(out_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer_video = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

    first_write = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if first_write:
            writer.writerow(["source", "frame", "x1", "y1", "x2", "y2",
                             "det_conf", "ocr_text", "ocr_conf", "crop_path", "vis_path"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            H, W = frame.shape[:2]

            results = model.predict(source=frame,
                                    conf=args.conf, iou=args.iou,
                                    device=device, verbose=False)
            r = results[0]

            annotated = r.plot()
            writer_video.write(annotated)

            if not r.boxes or len(r.boxes) == 0:
                writer.writerow([video_path, frame_idx, "", "", "", "",
                                 "", "", "", "", str(out_video)])
            else:
                for i, (xyxy, conf) in enumerate(zip(r.boxes.xyxy.cpu().numpy(),
                                                     r.boxes.conf.cpu().numpy())):
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1p, y1p, x2p, y2p = pad_and_clip_box(x1, y1, x2, y2,
                                                          args.pad, W, H)
                    crop = frame[y1p:y2p, x1p:x2p]

                    if crop.size == 0:
                        writer.writerow([video_path, frame_idx, x1, y1, x2, y2,
                                         float(conf), "", 0.0, "", str(out_video)])
                        continue

                    crop_for_ocr = crop if args.no_preproc else preprocess_for_ocr(crop)
                    ocr_results = reader.readtext(crop_for_ocr)
                    candidates = [(normalize_plate_text(text), float(c))
                                  for _, text, c in ocr_results if normalize_plate_text(text)]
                    best_text, best_conf = pick_best_candidate(candidates)

                    if best_conf < args.ocr_min_conf:
                        best_text = ""

                    crop_name = f"{Path(video_path).stem}_f{frame_idx}_plate{i}.jpg"
                    crop_out = crops_dir / crop_name
                    ensure_dir(crop_out)
                    cv2.imwrite(str(crop_out), crop)

                    writer.writerow([video_path, frame_idx, x1p, y1p, x2p, y2p,
                                     float(conf), best_text, best_conf,
                                     str(crop_out), str(out_video)])

            frame_idx += 1

    cap.release()
    writer_video.release()
    print(f"[OK] Processed video saved ➜ {out_video}")


def main():
    ap = argparse.ArgumentParser(description="License Plate Recognition (YOLOv8 + EasyOCR)")
    ap.add_argument("--weights", type=str, default="models/plate-detector.pt", help="YOLOv8 weights path")
    ap.add_argument("--source", type=str, required=True, help="Image/Folder/Video path")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold")
    ap.add_argument("--pad", type=float, default=0.06, help="Padding % for crops")
    ap.add_argument("--ocr_min_conf", type=float, default=0.30, help="Min OCR conf to accept")
    ap.add_argument("--out_dir", type=str, default="assets/outputs", help="Base output dir")
    ap.add_argument("--no_preproc", action="store_true", help="Disable OCR preprocessing")
    ap.add_argument("--device", type=str, default=None, help="force device: 'cpu' or '0'")
    args = ap.parse_args()

    source = Path(args.source).resolve()
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "vis"
    crops_dir = out_dir / "crops"
    csv_path = out_dir / "preds.csv"
    ensure_dir(csv_path)

    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")

    model = YOLO(args.weights)
    reader = easyocr.Reader(['en'], gpu=(device != "cpu"))

    src_type, data = collect_sources(source)

    if src_type == "images":
        process_images(data, model, reader, args, vis_dir, crops_dir, csv_path, device)
    elif src_type == "video":
        process_video(data, model, reader, args, vis_dir, crops_dir, csv_path, device)


if __name__ == "__main__":
    main()
