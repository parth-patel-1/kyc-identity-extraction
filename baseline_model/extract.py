import os
import sys
import re

from ultralytics import YOLO
from PIL import Image
import pytesseract
import pandas as pd  # pip install pandas openpyxl

# Map your YOLO class IDs to semantic labels
CLASSES = {
    0: 'Aadhar Number',
    1: 'DOB',
    2: 'Gender',
    3: 'Name',
    4: 'Address'
}

# If Tesseract is not on PATH, set this:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_aadhar_info(image_path: str,
                        model: YOLO,
                        conf: float = 0.5) -> dict:
    """
    Use YOLO to detect regions (name, gender, aadhar number, etc.),
    crop them, run OCR, and return structured info.
    """
    results = model.predict(image_path, imgsz=640, conf=conf)
    result = results[0]  # single image

    boxes_xyxy = result.boxes.xyxy.cpu().numpy().tolist()
    class_ids = result.boxes.cls.cpu().numpy().tolist()

    image = Image.open(image_path)

    info = {
        "file_name": os.path.basename(image_path),
        "name": None,
        "gender": None,
        "aadhar_number": None,
        "dob": None,
        "address": None,
    }

    for box, cls in zip(boxes_xyxy, class_ids):
        cls = int(cls)
        if cls not in CLASSES:
            continue

        label = CLASSES[cls]

        x1, y1, x2, y2 = map(int, box)
        # Add a small margin around the detected box
        crop = image.crop((x1 - 5, y1 - 5, x2 + 5, y2 + 5))

        text = pytesseract.image_to_string(crop, config="--psm 6")
        text_clean = text.replace("\n", " ").strip()
        print(f"[{info['file_name']}] [{label}] OCR: {text_clean}")

        if label == "Aadhar Number":
            # Extract 12-digit Aadhaar number (4-4-4 format or continuous)
            m = re.search(r"(\d{4}\s?\d{4}\s?\d{4})", text_clean)
            if m:
                info["aadhar_number"] = m.group(1).replace(" ", "")

        elif label == "Gender":
            t = text_clean.upper()
            if "MALE" in t:
                info["gender"] = "Male"
            elif "FEMALE" in t:
                info["gender"] = "Female"

        elif label == "Name":
            # Take first non-empty line as name
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines:
                info["name"] = lines[0]

        elif label == "DOB":
            # Simple date pattern (very loose)
            m = re.search(r"(\d{2}[\/\-]\d{2}[\/\-]\d{4})", text_clean)
            if m:
                info["dob"] = m.group(1)
            else:
                info["dob"] = text_clean

        elif label == "Address":
            info["address"] = text_clean

    return info


def process_folder(folder_path: str,
                   model_path: str = "aadhar_yolo_weights.pt",
                   output_path: str = "aadhar_extracted.xlsx",
                   conf: float = 0.5):
    # Load YOLO model once
    model = YOLO(model_path)

    image_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    rows = []

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(image_extensions):
            continue

        image_path = os.path.join(folder_path, file_name)
        print(f"Processing: {image_path}")
        try:
            info = extract_aadhar_info(image_path, model, conf=conf)
            rows.append(info)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    if not rows:
        print("No images processed or no data extracted.")
        return

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    print(f"Saved extracted data to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract.py <folder_path> [model_path] [output.xlsx]")
        sys.exit(1)

    folder_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "aadhar_yolo_weights.pt"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "aadhar_extracted.xlsx"

    if not os.path.isdir(folder_path):
        print(f"Folder does not exist: {folder_path}")
        sys.exit(1)

    process_folder(folder_path, model_path=model_path, output_path=output_path)


if __name__ == "__main__":
    main()
