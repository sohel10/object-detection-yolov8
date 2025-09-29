# lpr_utils.py
from pathlib import Path
import cv2
import numpy as np
import re

ALNUM_PLATE = re.compile(r"[A-Z0-9]{4,10}")  # generic plate filter (tweak if needed)

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def pad_and_clip_box(x1, y1, x2, y2, pad_pct, w, h):
    dx = int((x2 - x1) * pad_pct)
    dy = int((y2 - y1) * pad_pct)
    x1p = max(0, x1 - dx)
    y1p = max(0, y1 - dy)
    x2p = min(w - 1, x2 + dx)
    y2p = min(h - 1, y2 + dy)
    return x1p, y1p, x2p, y2p

def preprocess_for_ocr(crop, do_gray=True, upscale=2.0, denoise=True, binarize=True, sharpen=True, invert_if_needed=True):
    img = crop.copy()
    if do_gray and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # upscale
    if upscale and upscale != 1.0:
        new_w = max(1, int(img.shape[1] * upscale))
        new_h = max(1, int(img.shape[0] * upscale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # denoise
    if denoise:
        img = cv2.fastNlMeansDenoising(img, h=15)
    # binarize
    if binarize:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # sharpen
    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        img = cv2.filter2D(img, -1, kernel)
    # optional invert if background/foreground looks swapped
    if invert_if_needed:
        # heuristic: if mean is high (mostly white), keep; if mostly black, invert
        if np.mean(img) < 110:
            img = cv2.bitwise_not(img)
    return img

def normalize_plate_text(text):
    # upper, remove spaces and common separators
    t = text.upper().strip()
    t = t.replace(" ", "").replace("-", "").replace("_", "")
    # fix common OCR confusions (apply conservatively)
    t = t.replace("O", "0")  # O -> zero
    t = t.replace("I", "1")  # I -> one
    t = t.replace("B", "8")  # B -> eight (optional)
    t = t.replace("S", "5")  # S -> five (optional)
    # keep only alnum
    t = "".join(ch for ch in t if ch.isalnum())
    return t

def pick_best_candidate(candidates):
    """
    candidates: list of (text, conf)
    returns best_text, best_conf or ("", 0.0) if none look valid
    """
    if not candidates:
        return "", 0.0
    # prefer by confidence, then by regex match
    cand_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
    for txt, conf in cand_sorted:
        if ALNUM_PLATE.search(txt):
            return txt, conf
    return cand_sorted[0]  # fallback: best conf even if regex fails
