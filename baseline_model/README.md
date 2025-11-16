# NetraAadhar- Baseline model
NetraAdhar Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10975021 

NetraAadhar Repo: https://github.com/Adinp1213/NetraAadhar

A Python-based pipeline to automatically extract key fields from Aadhaar card images using a YOLO model for field detection and Tesseract OCR for text recognition. Supports:

- Single-image inference for debugging.
- Batch inference on a folder of images, exporting all fields to an Excel (`.xlsx`) file.
- (Optional) real-time / webcam inference via `realtime.py`.
- QR code extraction via `qr_code.py` (if configured).

---

## 1. Installation

### 1.1. Python dependencies

Create and activate a virtual environment (recommended), then install:

pip install -r requirements.txt

`requirements.txt` includes:

- ultralytics – YOLOv8 model and inference.
- pytesseract – Python wrapper for Tesseract OCR.
- opencv-python – Image/video utilities (e.g. webcam).
- pandas, openpyxl – For Excel file creation.
- numpy, Pillow – Image processing.
- django – For any web UI endpoints you may add.

### 1.2. System Tesseract OCR

You MUST install Tesseract OCR as a separate system program:

- Windows  
  Download and install from: https://github.com/UB-Mannheim/tesseract/wiki  
  Then add the Tesseract install folder (e.g. C:\Program Files\Tesseract-OCR) to your PATH.

- Linux (Debian/Ubuntu)

  sudo apt-get install tesseract-ocr

- macOS (Homebrew)

  brew install tesseract

If Tesseract is not on your PATH, set the path manually in your code:

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

---

## 2. Project Structure (typical)

.
├── main.py                 # Main entrypoint (if used)
├── realtime.py             # Real-time / webcam inference
├── text_extraction.py      # Single-image YOLO + OCR example
├── extract.py              # Batch inference: folder → Excel
├── qr_code.py              # QR code extraction (if used)
├── aadhar_yolo_weights.pt  # YOLOv8 trained weights for Aadhaar fields
├── README.md
└── requirements.txt

---

## 3. YOLO + Tesseract Field Extraction

The YOLO model is trained to detect specific regions on the Aadhaar card:

- Class 0 – Aadhaar Number
- Class 1 – Date of Birth (DOB)
- Class 2 – Gender
- Class 3 – Name
- Class 4 – Address

For each detected bounding box:

1. The image is cropped around the box.
2. The cropped region is passed to Tesseract OCR.
3. Simple parsing / regex is applied to:
   - Normalize the Aadhaar number (12 digits).
   - Infer Gender (Male / Female).
   - Parse DOB where possible.
   - Extract Name and Address text.

---

## 4. Batch Inference (Folder → Excel) – Detailed Usage

This is handled by extract.py (the batch script). It:

- Takes a folder path as input.
- Loads the YOLO model once.
- Runs inference on each image in the folder.
- Extracts fields via YOLO + Tesseract.
- Stores all results in a single .xlsx file.



### 4.2. Running batch inference

1. Place all your Aadhaar images in a folder.

2. Ensure your YOLO weights file is available (e.g. aadhar_yolo_weights.pt in the project root).

3. Run the batch script from the project directory:

   ```python extract.py <folder path> aadhar_yolo_weights.pt output.xlsx```

   Arguments:
   - extract.py – batch script.
   - folder_path – first argument (required).
   - model_path – second argument (optional, defaults to aadhar_yolo_weights.pt).
   - output.xlsx – third argument (optional, defaults to aadhar_extracted.xlsx).

4. For each image, you’ll see console logs like:

   Processing: D:/.../adhar_ (3).jpg
   [adhar_ (3).jpg] [Name] OCR: JOHN DOE
   [adhar_ (3).jpg] [Gender] OCR: MALE
   [adhar_ (3).jpg] [Aadhar Number] OCR: 1234 5678 9012
   ...

5. When finished, open the generated Excel file:

   aadhar_extracted.xlsx

   The sheet will typically contain columns like:

   - file_name
   - name
   - gender
   - aadhar_number
   - dob
   - address

Each row corresponds to a single image in the input folder.

---

## 5. Single-Image Inference (Debugging)

You can experiment with single-image inference using text_extraction.py or a similar script:

```python text_extraction.py```

Typical workflow inside the script:

1. Set large_image_path to your Aadhaar image.
2. Run YOLO detection (model.track / model.predict).
3. For each detected box, crop the bounding box and run Tesseract OCR.
4. Print the extracted text per field.

This is useful for debugging predictions before running batch inference.

