import os
import re
from typing import Dict, Any, List

from PIL import Image
from tqdm import tqdm
import pandas as pd

from app_gradio_kyc import (
    EXTRACTOR,
    PROMPT_AADHAAR_FRONT,
    PROMPT_AADHAAR_BACK,
    PROMPT_PAN,
    canonicalize_keys,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _collect_image_files(input_dir: str) -> List[str]:
    files: List[str] = []
    for fname in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, fname)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMG_EXTS:
            continue
        files.append(fname)
    return files


# ---------- Shared helpers ----------

def clean_aadhaar_number(value: str) -> str:
    # Remove spaces first, then keep only digits
    value = str(value).replace(" ", "")
    digits = re.sub(r"\D", "", value)
    return digits[:12] if len(digits) >= 12 else digits


def _strip_dob_from_text(text: str) -> str:
    """
    Remove any DOB-like fragments (e.g. 'DOB: 01/01/1990') from the text.
    This keeps address clean for Aadhaar BACK side where the
    model may occasionally mix DOB into the address field.
    """
    if not text:
        return ""

    # Pattern for common date formats like 01/01/1990 or 1-1-90
    date_pattern = r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"

    # Remove phrases like "DOB: 01/01/1990", "Date of Birth - 01-01-1990", etc.
    text = re.sub(
        rf"(?i)\b(?:dob|d\.o\.b|date of birth)\b[:\-\s]*{date_pattern}",
        "",
        text,
    )

    # If the word DOB is still present, remove nearby date tokens and the word itself
    if re.search(r"(?i)\b(dob|d\.o\.b|date of birth)\b", text):
        text = re.sub(date_pattern, "", text)
        text = re.sub(r"(?i)\b(dob|d\.o\.b|date of birth)\b[:\-]?", "", text)

    # Collapse extra spaces and trim punctuation
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,;:-")


# ---------- Aadhaar FRONT ----------

def normalize_front_fields(raw: Dict[str, Any]) -> Dict[str, str]:
    """
    Map the model JSON into: name, gender, dob, aadhaar_number
    (same behaviour as previous batch_infer.py)
    """
    out = {"name": "", "gender": "", "dob": "", "aadhaar_number": ""}

    if not isinstance(raw, dict):
        return out

    for k, v in raw.items():
        if v is None:
            continue
        text = str(v).strip()
        lk = re.sub(r"[^a-z]", "", str(k).lower())

        if lk.startswith("name") or "fullname" in lk:
            out["name"] = text
        elif "gender" in lk or "sex" in lk:
            out["gender"] = text
        elif "dob" in lk or "birth" in lk:
            out["dob"] = text
        elif "aadhaar" in lk or "aadhar" in lk or lk == "uid":
            num = clean_aadhaar_number(text)
            out["aadhaar_number"] = num or text

    return out


def run_aadhaar_front_folder(input_dir: str, output_xlsx: str, batch_size: int = 4) -> str:
    """
    Batch Aadhaar FRONT OCR.
    Output is a real Excel file (.xlsx) with columns:
    file, name, gender, dob, aadhaar_number
    """
    rows: List[Dict[str, Any]] = []
    file_list = _collect_image_files(input_dir)

    with tqdm(total=len(file_list), desc="Processing Aadhaar FRONT images") as pbar:
        for i in range(0, len(file_list), batch_size):
            batch_fnames = file_list[i : i + batch_size]

            batch_imgs: List[Image.Image] = []
            valid_fnames: List[str] = []

            for fname in batch_fnames:
                path = os.path.join(input_dir, fname)
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    continue
                batch_imgs.append(img)
                valid_fnames.append(fname)

            if not batch_imgs:
                pbar.update(len(batch_fnames))
                continue

            raw_list = EXTRACTOR.extract_batch(batch_imgs, PROMPT_AADHAAR_FRONT)

            for fname, raw in zip(valid_fnames, raw_list):
                fields = normalize_front_fields(raw)
                fields["file"] = fname
                rows.append(fields)

            pbar.update(len(batch_fnames))

    fieldnames = ["file", "name", "gender", "dob", "aadhaar_number"]
    df = pd.DataFrame(rows, columns=fieldnames)
    os.makedirs(os.path.dirname(os.path.abspath(output_xlsx)), exist_ok=True)
    df.to_excel(output_xlsx, index=False)
    return os.path.abspath(output_xlsx)


# ---------- Aadhaar BACK ----------

def normalize_address(raw: Dict[str, Any]) -> str:
    if not isinstance(raw, dict):
        return ""
    for k, v in raw.items():
        if v is None:
            continue
        lk = re.sub(r"[^a-z]", "", str(k).lower())
        if "address" in lk:
            return _strip_dob_from_text(str(v).strip())
    return ""


def normalize_aadhaar_number(raw: Dict[str, Any]) -> str:
    if not isinstance(raw, dict):
        return ""
    for k, v in raw.items():
        if v is None:
            continue
        lk = re.sub(r"[^a-z]", "", str(k).lower())
        if "aadhaar" in lk or "aadhar" in lk or lk == "uid":
            num = clean_aadhaar_number(str(v))
            return num or str(v).strip()
    return ""


def run_aadhaar_back_folder(
    input_dir: str,
    output_xlsx: str,
    batch_size: int = 4,
    use_front_prompt_for_aadhaar: bool = False,
) -> str:
    """
    input_dir: folder containing Aadhaar BACK images.
    output_xlsx: path to .xlsx file to write.
    If use_front_prompt_for_aadhaar=True, we run a second pass
    with the FRONT prompt just to get aadhaar_number more reliably.
    """
    rows: List[Dict[str, Any]] = []
    file_list = _collect_image_files(input_dir)

    with tqdm(total=len(file_list), desc="Processing Aadhaar BACK images") as pbar:
        for i in range(0, len(file_list), batch_size):
            batch_fnames = file_list[i : i + batch_size]

            batch_imgs: List[Image.Image] = []
            valid_fnames: List[str] = []

            for fname in batch_fnames:
                path = os.path.join(input_dir, fname)
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    continue
                batch_imgs.append(img)
                valid_fnames.append(fname)

            if not batch_imgs:
                pbar.update(len(batch_fnames))
                continue

            # First pass: back prompt to get address
            raw_back_list = EXTRACTOR.extract_batch(batch_imgs, PROMPT_AADHAAR_BACK)

            # Optional second pass: front prompt to get Aadhaar number
            if use_front_prompt_for_aadhaar:
                raw_front_list = EXTRACTOR.extract_batch(batch_imgs, PROMPT_AADHAAR_FRONT)
            else:
                raw_front_list = [None] * len(batch_imgs)

            for fname, raw_back, raw_front in zip(valid_fnames, raw_back_list, raw_front_list):
                address = normalize_address(raw_back)
                # Prefer Aadhaar number from the FRONT-style prompt if requested,
                # otherwise fall back to whatever the BACK prompt returns.
                aadhaar_number = ""
                if raw_front is not None:
                    aadhaar_number = normalize_aadhaar_number(raw_front)
                if not aadhaar_number:
                    aadhaar_number = normalize_aadhaar_number(raw_back)

                rows.append(
                    {
                        "file": fname,
                        "aadhaar_number": aadhaar_number,
                        "address": address,
                    }
                )

            pbar.update(len(batch_fnames))

    df = pd.DataFrame(rows, columns=["file", "aadhaar_number", "address"])
    os.makedirs(os.path.dirname(os.path.abspath(output_xlsx)), exist_ok=True)
    df.to_excel(output_xlsx, index=False)
    return os.path.abspath(output_xlsx)


# ---------- PAN ----------

def run_pan_folder(input_dir: str, output_xlsx: str, batch_size: int = 4) -> str:
    """
    Batch PAN OCR using the PROMPT_PAN and canonicalize_keys
    from app_gradio_kyc.
    Outputs: file, permanent_account_number, name, father_s_name, date_of_birth
    """
    rows: List[Dict[str, Any]] = []
    file_list = _collect_image_files(input_dir)

    with tqdm(total=len(file_list), desc="Processing PAN images") as pbar:
        for i in range(0, len(file_list), batch_size):
            batch_fnames = file_list[i : i + batch_size]

            batch_imgs: List[Image.Image] = []
            valid_fnames: List[str] = []

            for fname in batch_fnames:
                path = os.path.join(input_dir, fname)
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    continue
                batch_imgs.append(img)
                valid_fnames.append(fname)

            if not batch_imgs:
                pbar.update(len(batch_fnames))
                continue

            raw_list = EXTRACTOR.extract_batch(batch_imgs, PROMPT_PAN)

            for fname, raw in zip(valid_fnames, raw_list):
                norm = canonicalize_keys(raw)
                row = {
                    "file": fname,
                    "permanent_account_number": norm.get("permanent_account_number", ""),
                    "name": norm.get("name", ""),
                    "father_s_name": norm.get("father_s_name", ""),
                    "date_of_birth": norm.get("DOB") or norm.get("date_of_birth", ""),
                }
                rows.append(row)

            pbar.update(len(batch_fnames))

    fieldnames = ["file", "permanent_account_number", "name", "father_s_name", "date_of_birth"]
    df = pd.DataFrame(rows, columns=fieldnames)
    os.makedirs(os.path.dirname(os.path.abspath(output_xlsx)), exist_ok=True)
    df.to_excel(output_xlsx, index=False)

    return os.path.abspath(output_xlsx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified batch OCR for Aadhaar FRONT, Aadhaar BACK, and PAN"
    )

    subparsers = parser.add_subparsers(dest="doc_type", required=True)

    # Aadhaar front
    p_front = subparsers.add_parser("aadhaar_front", help="Batch Aadhaar FRONT OCR to XLSX")
    p_front.add_argument("folder", help="Folder with Aadhaar front images")
    p_front.add_argument(
        "-o", "--output", default="aadhaar_front.xlsx", help="Output XLSX path"
    )
    p_front.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Number of images per GPU batch",
    )

    # Aadhaar back
    p_back = subparsers.add_parser("aadhaar_back", help="Batch Aadhaar BACK OCR to XLSX")
    p_back.add_argument("folder", help="Folder with Aadhaar back images")
    p_back.add_argument(
        "-o",
        "--output",
        default="aadhaar_back.xlsx",
        help="Output XLSX path",
    )
    p_back.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Number of images per GPU batch",
    )
    p_back.add_argument(
        "--use-front-prompt-for-aadhaar",
        action="store_true",
        help="Also run FRONT prompt on back images to extract Aadhaar number",
    )

    # PAN
    p_pan = subparsers.add_parser("pan", help="Batch PAN OCR to XLSX")
    p_pan.add_argument("folder", help="Folder with PAN images")
    p_pan.add_argument(
        "-o", "--output", default="pan.xlsx", help="Output XLSX path"
    )
    p_pan.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Number of images per GPU batch",
    )

    args = parser.parse_args()

    if args.doc_type == "aadhaar_front":
        out_path = run_aadhaar_front_folder(
            args.folder, args.output, batch_size=args.batch_size
        )
    elif args.doc_type == "aadhaar_back":
        out_path = run_aadhaar_back_folder(
            args.folder,
            args.output,
            batch_size=args.batch_size,
            use_front_prompt_for_aadhaar=args.use_front_prompt_for_aadhaar,
        )
    elif args.doc_type == "pan":
        out_path = run_pan_folder(
            args.folder, args.output, batch_size=args.batch_size
        )
    else:
        parser.error("Unknown doc_type")

    print(f"Saved output to: {out_path}")
