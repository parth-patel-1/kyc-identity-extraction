import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from typing import Dict, Any, Tuple, List
import io, json, re, unicodedata, tempfile
from datetime import datetime

import gradio as gr
from PIL import Image
import torch
from rapidfuzz import fuzz
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

from text_scoring import (
    score_exact_string,
    score_exact_digits,
    score_exact_date,
    score_proportional_words,
)

# ====================== MODEL (LOAD AT START) ======================

def device_dtype():
    has_cuda = torch.cuda.is_available()
    return ("cuda", torch.float16) if has_cuda else ("cpu", "auto")

def resize_if_tall(img: Image.Image, target_h: int = 500) -> Image.Image:
    w, h = img.size
    if h <= target_h:
        return img
    new_w = int(target_h * (w / h))
    return img.resize((new_w, target_h))

def save_temp_image(img: Image.Image, suffix=".jpg") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path

def safe_json_from_text(txt: str) -> Dict[str, Any]:
    txt = txt.replace("```", "").replace("json", "").strip()
    try:
        return json.loads(txt)
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(txt[s:e+1])
        return {"raw_text": txt}

class OCRExtractor:
    def __init__(self, model_path: str = "nanonets/Nanonets-OCR-s"):
        device, dtype = device_dtype()
        self.has_cuda = (device == "cuda")
        self.device = device

        # Explicit fast tokenizer/processor to match your request
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if self.has_cuda else None,
            attn_implementation="eager",
        )
        if not self.has_cuda:
            self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract_batch(
        self,
        imgs: List[Image.Image],
        prompt: str,
        max_new_tokens: int = 512,
    ) -> List[Dict[str, Any]]:
        if not imgs:
            return []

        resized_imgs: List[Image.Image] = []
        tmp_paths: List[str] = []
        texts: List[str] = []

        for img in imgs:
            img_rs = resize_if_tall(img, 500)
            tmp_path = save_temp_image(img_rs)
            abspath = os.path.abspath(tmp_path)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{abspath}"},
                    {"type": "text",  "text": f"{prompt} Return strict JSON with only the requested fields."},
                ]},
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            texts.append(text)
            resized_imgs.append(img_rs)
            tmp_paths.append(tmp_path)

        inputs = self.processor(
            text=texts,
            images=resized_imgs,
            padding=True,
            return_tensors="pt",
        )

        if self.has_cuda:
            torch.cuda.empty_cache()
            inputs = inputs.to(self.model.device)

        try:
            autocast = torch.cuda.amp.autocast if self.has_cuda else torch.cpu.amp.autocast
        except AttributeError:
            class _NoCtx:
                def __enter__(self): return None
                def __exit__(self, *a): return False

            def autocast(*a, **k): return _NoCtx()

        try:
            with autocast(enabled=self.has_cuda):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        except torch.cuda.OutOfMemoryError:
            if self.has_cuda:
                torch.cuda.empty_cache()
                self.model.to("cpu")
                inputs = inputs.to("cpu")
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
            else:
                raise

        gen_ids = [
            out_ids[len(inp):]
            for inp, out_ids in zip(inputs.input_ids, output_ids)
        ]

        decoded = self.processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for tmp in tmp_paths:
            try:
                os.remove(tmp)
            except Exception:
                pass

        return [safe_json_from_text(txt.strip()) for txt in decoded]

    @torch.inference_mode()
    def extract(self, img: Image.Image, prompt: str, max_new_tokens: int = 512) -> Dict[str, Any]:
        results = self.extract_batch([img], prompt, max_new_tokens=max_new_tokens)
        return results[0] if results else {}

# >>> Load model immediately on script start (Change #1)
EXTRACTOR = OCRExtractor()

# ====================== PROMPTS ======================
PROMPT_AADHAAR_FRONT = "Extract the english text of name, DOB, gender and aadhar number, in json format with keys: name, DOB, gender, aadhar_number."
PROMPT_AADHAAR_BACK  = "Extract the english text of address and aadhar number, in json format with keys: address, aadhar_number."
PROMPT_PAN           = "Extract the text of permanent account number, name, father's name and date of birth, in json format with keys: permanent_account_number, name, father_s_name, date_of_birth."

# ====================== NORMALIZATION & SCORING ======================

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_text(s: str) -> str:
    s = strip_accents(s or "")
    s = s.replace("\u200b", "").replace("\xa0", " ")
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_name(s: str) -> str:
    return normalize_text(s).upper()

def normalize_gender(s: str) -> str:
    s = (s or "").strip().lower()
    if s in {"m", "male"}: return "male"
    if s in {"f", "female"}: return "female"
    if s in {"o", "other", "others"}: return "other"
    return s

def normalize_aadhaar(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def normalize_pan(s: str) -> str:
    return normalize_text(s).upper().replace(" ", "")

def normalize_address(s: str) -> str:
    return normalize_text(s).upper()

def normalize_date(s: str) -> str:
    s = (s or "").strip().replace("\\", "/").replace("-", "/").replace(".", "/")
    s = re.sub(r"\s+", "", s)
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if m:
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100: y += 2000 if y <= 30 else 1900
        try:
            return datetime(y, mth, d).strftime("%Y-%m-%d")
        except ValueError:
            pass
    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", s)
    if m:
        y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(y, mth, d).strftime("%Y-%m-%d")
        except ValueError:
            pass
    digits = re.sub(r"\D", "", s)
    if len(digits) == 8:
        d, mth, y = int(digits[:2]), int(digits[2:4]), int(digits[4:])
        try:
            return datetime(y, mth, d).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return s

def canonicalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {"raw": d}
    kv = { re.sub(r"\s+", "_", normalize_text(str(k)).lower()): v for k, v in d.items() }
    out = {}
    for key in ["name","full_name"]:
        if key in kv: out["name"] = kv[key]
    for key in ["dob","date_of_birth","d_o_b"]:
        if key in kv: out["DOB"] = kv[key]
    for key in ["gender","sex"]:
        if key in kv: out["gender"] = kv[key]
    for key in ["aadhar_number","aadhaar_number","aadhar_no","aadhaar_no","uid","uidai"]:
        if key in kv: out["aadhar_number"] = kv[key]
    for key in ["address","residential_address","addr"]:
        if key in kv: out["address"] = kv[key]
    for key in ["permanent_account_number","pan","pan_number","pan_no"]:
        if key in kv: out["permanent_account_number"] = kv[key]
    for key in ["father_s_name","fathers_name","father_name"]:
        if key in kv: out["father_s_name"] = kv[key]
    if "date_of_birth" in kv and "DOB" not in out:
        out["date_of_birth"] = kv["date_of_birth"]
    return out

def score_exact_or_fuzzy(a: str, b: str, norm_func, allow_fuzzy=True) -> Tuple[int, str]:
    if not a or not b:
        return 0, "Missing value"
    na, nb = norm_func(a), norm_func(b)
    if not na or not nb:
        return 0, "Empty after normalization"
    if na == nb:
        return 100, "Exact match"
    if not allow_fuzzy:
        return 0, "Mismatch (no fuzzy)"
    r = fuzz.token_set_ratio(na, nb)
    return int(r), f"Fuzzy similarity {r}"

def score_gender(a: str, b: str) -> Tuple[int, str]:
    ga, gb = normalize_gender(a or ""), normalize_gender(b or "")
    if not ga or not gb: return 0, "Missing value"
    return (100, "Exact match") if ga == gb else (0, f"Mismatch ({ga} vs {gb})")

def score_date(a: str, b: str) -> Tuple[int, str]:
    if not a or not b: return 0, "Missing value"
    na, nb = normalize_date(a), normalize_date(b)
    if na == nb and re.match(r"^\d{4}-\d{2}-\d{2}$", na):
        return 100, "Exact normalized date match"
    da, db = re.sub(r"\D", "", na), re.sub(r"\D", "", nb)
    if da == db and len(da) >= 6:
        return 90, "Digits-only date match"
    return 0, f"Mismatch ({na} vs {nb})"

def severity_from_score(overall: int) -> str:
    if overall >= 90: return "LOW"
    if overall >= 70: return "MEDIUM"
    return "HIGH"

def aggregate_score(field_scores: Dict[str, int]) -> int:
    weights = {"name":0.25,"dob":0.20,"address":0.20,"pan_number":0.15,"aadhaar_number":0.15,"gender":0.05}
    total = 0.0
    for k, w in weights.items():
        if k in field_scores and field_scores[k] is not None:
            total += w * field_scores[k]
    return int(round(total))

# ====================== PIPELINE (images optional) ======================

def run_pipeline(
    selected_docs: List[str],
    aadhaar_front_img: Image.Image | None,
    aadhaar_back_img: Image.Image | None,
    pan_img: Image.Image | None,
    typed_name: str,
    typed_dob: str,
    typed_gender: str,
    typed_aadhaar: str,
    typed_address: str,
    typed_pan: str,
    typed_pan_name: str,
    typed_father: str,
):
    extracted = {}

    # Extract only what the user selected & uploaded
    if "Aadhaar Front" in selected_docs and aadhaar_front_img is not None:
        af_ex = canonicalize_keys(EXTRACTOR.extract(aadhaar_front_img, PROMPT_AADHAAR_FRONT, 512))
        extracted["aadhaar_front"] = af_ex

    if "Aadhaar Back" in selected_docs and aadhaar_back_img is not None:
        ab_ex = canonicalize_keys(EXTRACTOR.extract(aadhaar_back_img, PROMPT_AADHAAR_BACK, 256))
        extracted["aadhaar_back"] = ab_ex

    if "PAN" in selected_docs and pan_img is not None:
        pn_ex = canonicalize_keys(EXTRACTOR.extract(pan_img, PROMPT_PAN, 512))
        extracted["pan"] = pn_ex

    # Build comparisons only for available extracted data
    results = {}
    # Name
    name_candidates = []
    if "aadhaar_front" in extracted and extracted["aadhaar_front"].get("name"):
        s, r = score_proportional_words(typed_name, extracted["aadhaar_front"]["name"])
        name_candidates.append(("Aadhaar", s, r))
    if "pan" in extracted and extracted["pan"].get("name"):
        s, r = score_proportional_words(typed_name, extracted["pan"]["name"])
        name_candidates.append(("PAN", s, r))
    if name_candidates:
        best = max(name_candidates, key=lambda x: x[1])
        results["name"] = {"score": best[1], "explanations": [f"{src}: {reason}" for src,_,reason in name_candidates]}
    else:
        results["name"] = {"score": None, "explanations": ["Skipped (no extracted name available)"]}

    # DOB (Aadhaar vs PAN consistency + form)
    aadhaar_dob = extracted.get("aadhaar_front", {}).get("DOB")
    pan_dob = extracted.get("pan", {}).get("date_of_birth")

    # If both documents have DOB and they disagree, force score to 0
    if aadhaar_dob and pan_dob:
        na = normalize_date(str(aadhaar_dob))
        np = normalize_date(str(pan_dob))
        if na and np and na != np:
            results["dob"] = {
                "score": 0,
                "explanations": [
                    f"Aadhaar DOB: {aadhaar_dob}",
                    f"PAN DOB: {pan_dob}",
                    "Mismatch between Aadhaar and PAN DOB \u21d2 score forced to 0",
                ],
            }

    if "dob" not in results:
        dob_candidates = []
        if aadhaar_dob:
            s, r = score_exact_date(typed_dob, aadhaar_dob)
            dob_candidates.append(("Aadhaar", s, r))
        if pan_dob:
            s, r = score_exact_date(typed_dob, pan_dob)
            dob_candidates.append(("PAN", s, r))
        if dob_candidates:
            best = max(dob_candidates, key=lambda x: x[1])
            results["dob"] = {
                "score": best[1],
                "explanations": [f"{src}: {reason}" for src, _, reason in dob_candidates],
            }
        else:
            results["dob"] = {"score": None, "explanations": ["Skipped (no extracted DOB available)"]}

    # Gender (Aadhaar Front)
    if "aadhaar_front" in extracted and extracted["aadhaar_front"].get("gender"):
        s, r = score_exact_string(typed_gender, extracted["aadhaar_front"]["gender"])
        results["gender"] = {"score": s, "explanations": [r]}
    else:
        results["gender"] = {"score": None, "explanations": ["Skipped (no extracted gender available)"]}

    # Aadhaar number (front/back consistency + form)
    front_num = extracted.get("aadhaar_front", {}).get("aadhar_number")
    back_num = extracted.get("aadhaar_back", {}).get("aadhar_number")
    if front_num or back_num:
        if front_num and back_num:
            fn_norm = normalize_aadhaar(front_num)
            bn_norm = normalize_aadhaar(back_num)
            if fn_norm and bn_norm and fn_norm != bn_norm:
                results["aadhaar_number"] = {
                    "score": 0,
                    "explanations": [
                        f"Front Aadhaar: {front_num}",
                        f"Back Aadhaar: {back_num}",
                        "Mismatch between front and back Aadhaar numbers ⇒ score forced to 0",
                    ],
                }
            else:
                ref_num = front_num or back_num
                s, r = score_exact_digits(typed_aadhaar, ref_num)
                expl = [
                    f"Front Aadhaar: {front_num}" if front_num else "Front Aadhaar: N/A",
                    f"Back Aadhaar: {back_num}" if back_num else "Back Aadhaar: N/A",
                    r,
                ]
                results["aadhaar_number"] = {"score": s, "explanations": expl}
        else:
            ref_num = front_num or back_num
            s, r = score_exact_digits(typed_aadhaar, ref_num)
            src = "Aadhaar Front" if front_num else "Aadhaar Back"
            results["aadhaar_number"] = {"score": s, "explanations": [f"{src}: {r}"]}
    else:
        results["aadhaar_number"] = {"score": None, "explanations": ["Skipped (no extracted Aadhaar number available)"]}

    # Address (Aadhaar Back)
    if "aadhaar_back" in extracted and extracted["aadhaar_back"].get("address"):
        s, r = score_proportional_words(typed_address, extracted["aadhaar_back"]["address"])
        results["address"] = {"score": s, "explanations": [r]}
    else:
        results["address"] = {"score": None, "explanations": ["Skipped (no extracted address available)"]}

    # PAN number
    if "pan" in extracted and extracted["pan"].get("permanent_account_number"):
        s, r = score_exact_digits(typed_pan, extracted["pan"]["permanent_account_number"])
        results["pan_number"] = {"score": s, "explanations": [r]}
    else:
        results["pan_number"] = {"score": None, "explanations": ["Skipped (no extracted PAN number available)"]}

    # PAN name (form vs PAN document)
    if "pan" in extracted and extracted["pan"].get("name"):
        s, r = score_proportional_words(typed_pan_name, extracted["pan"]["name"])
        results["pan_name"] = {"score": s, "explanations": [r]}
    else:
        results["pan_name"] = {"score": None, "explanations": ["Skipped (no extracted PAN name available)"]}

    # Father's name
    if "pan" in extracted and extracted["pan"].get("father_s_name"):
        s, r = score_proportional_words(typed_father, extracted["pan"]["father_s_name"])
        results["father_name"] = {"score": s, "explanations": [r]}
    else:
        results["father_name"] = {"score": None, "explanations": ["Skipped (no extracted father's name available)"]}

    # Overall (weights) only over available fields
    field_scores = {
        "name": results["name"]["score"],
        "dob": results["dob"]["score"],
        "address": results["address"]["score"],
        "pan_number": results["pan_number"]["score"],
        "aadhaar_number": results["aadhaar_number"]["score"],
        "gender": results["gender"]["score"],
    }
    # Remove None
    field_scores = {k:v for k,v in field_scores.items() if v is not None}
    overall = aggregate_score(field_scores) if field_scores else 0
    severity = severity_from_score(overall) if field_scores else "N/A"

    # Prepare typed vs extracted display strings
    def _join(parts):
        return "<br>".join(parts) if parts else "N/A"

    name_extracted_parts = []
    if "aadhaar_front" in extracted and extracted["aadhaar_front"].get("name"):
        name_extracted_parts.append(f"Aadhaar: {extracted['aadhaar_front']['name']}")
    if "pan" in extracted and extracted["pan"].get("name"):
        name_extracted_parts.append(f"PAN: {extracted['pan']['name']}")
    name_extracted_display = _join(name_extracted_parts)

    dob_extracted_parts = []
    if "aadhaar_front" in extracted and extracted["aadhaar_front"].get("DOB"):
        dob_extracted_parts.append(f"Aadhaar: {extracted['aadhaar_front']['DOB']}")
    if "pan" in extracted and extracted["pan"].get("date_of_birth"):
        dob_extracted_parts.append(f"PAN: {extracted['pan']['date_of_birth']}")
    dob_extracted_display = _join(dob_extracted_parts)

    if "aadhaar_front" in extracted and extracted["aadhaar_front"].get("gender"):
        gender_extracted_display = extracted["aadhaar_front"]["gender"]
    else:
        gender_extracted_display = "N/A"

    aadhaar_extracted_parts = []
    if "aadhaar_front" in extracted and extracted["aadhaar_front"].get("aadhar_number"):
        aadhaar_extracted_parts.append(f"Front: {extracted['aadhaar_front']['aadhar_number']}")
    if "aadhaar_back" in extracted and extracted["aadhaar_back"].get("aadhar_number"):
        aadhaar_extracted_parts.append(f"Back: {extracted['aadhaar_back']['aadhar_number']}")
    aadhaar_extracted_display = _join(aadhaar_extracted_parts)

    if "aadhaar_back" in extracted and extracted["aadhaar_back"].get("address"):
        address_extracted_display = extracted["aadhaar_back"]["address"]
    else:
        address_extracted_display = "N/A"

    if "pan" in extracted and extracted["pan"].get("permanent_account_number"):
        pan_extracted_display = extracted["pan"]["permanent_account_number"]
    else:
        pan_extracted_display = "N/A"

    if "pan" in extracted and extracted["pan"].get("name"):
        pan_name_extracted_display = extracted["pan"]["name"]
    else:
        pan_name_extracted_display = "N/A"

    if "pan" in extracted and extracted["pan"].get("father_s_name"):
        father_extracted_display = extracted["pan"]["father_s_name"]
    else:
        father_extracted_display = "N/A"

    # HTML report
    def badge(score):
        if score is None: return '<span style="padding:2px 8px;border-radius:9999px;background:#6b7280;color:white">N/A</span>'
        if score >= 90:   color = "#16a34a"
        elif score >= 70: color = "#f59e0b"
        else:             color = "#dc2626"
        return f'<span style="padding:2px 8px;border-radius:9999px;background:{color};color:white;font-weight:600">{score}</span>'

    def row(label, item, typed_value, extracted_value):
        return (
            "<tr>"
            f"<td><b>{label}</b></td>"
            f"<td>{typed_value or ''}</td>"
            f"<td>{extracted_value}</td>"
            f"<td style='text-align:center'>{badge(item['score'])}</td>"
            f"<td>{'<br>'.join(item['explanations'])}</td>"
            "</tr>"
        )

    html = f"""
    <div style="font-family:Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; line-height:1.45">
      <h2>KYC Consistency Report</h2>
      <p><b>Overall Consistency:</b> {badge(overall)} &nbsp; <b>Severity:</b> {severity}</p>
      <table style="border-collapse:collapse;width:100%">
        <thead><tr style="text-align:left;border-bottom:1px solid #eee">
          <th>Field</th>
          <th>Typed (Form)</th>
          <th>Extracted from Document(s)</th>
          <th style="text-align:center">Score</th>
          <th>Explanation</th>
        </tr></thead>
        <tbody>
          {row("Name (best of Aadhaar/PAN)", results["name"], typed_name, name_extracted_display)}
          {row("Date of Birth (best of Aadhaar/PAN)", results["dob"], typed_dob, dob_extracted_display)}
          {row("Gender (Aadhaar)", results["gender"], typed_gender, gender_extracted_display)}
          {row("Aadhaar Number", results["aadhaar_number"], typed_aadhaar, aadhaar_extracted_display)}
          {row("Address (Aadhaar Back)", results["address"], typed_address, address_extracted_display)}
          {row("PAN Number", results["pan_number"], typed_pan, pan_extracted_display)}
          {row("PAN Name", results["pan_name"], typed_pan_name, pan_name_extracted_display)}
          {row("Father's Name (PAN)", results["father_name"], typed_father, father_extracted_display)}
        </tbody>
      </table>
    </div>
    """

    payload = {
        "selected_docs": selected_docs,
        "typed_form": {
            "name": typed_name, "date_of_birth": typed_dob, "gender": typed_gender,
            "aadhaar_number": typed_aadhaar, "address": typed_address,
            "pan_number": typed_pan, "pan_name": typed_pan_name, "father_name": typed_father,
        },
        "extracted": extracted,
        "scores": results,
        "overall_score": overall,
        "severity": severity,
    }
    return payload, html

# ====================== GRADIO UI (images optional) ======================

with gr.Blocks(title="Document Consistency Checker (Aadhaar + PAN)") as demo:
    gr.Markdown("# Document Consistency Checker\nChoose which documents you’ll upload, then provide the images and typed details for those only.")

    doc_select = gr.CheckboxGroup(
        choices=["Aadhaar Front", "Aadhaar Back", "PAN"],
        value=["Aadhaar Front","Aadhaar Back", "PAN"],  # default selection
        label="Which documents will you upload?",
    )

    with gr.Row():
        aadhaar_front_img = gr.Image(label="Aadhaar Front", sources=["upload"], type="pil", visible=True)
        aadhaar_back_img  = gr.Image(label="Aadhaar Back",  sources=["upload"], type="pil", visible=False)
        pan_img           = gr.Image(label="PAN",            sources=["upload"], type="pil", visible=False)

    with gr.Row():
        name   = gr.Textbox(label="Full Name (form)", visible=True)
        dob    = gr.Textbox(label="Date of Birth (DD/MM/YYYY)", visible=True)
        gender = gr.Dropdown(choices=["Male","Female","Other"], value="Male", label="Gender (form)", visible=True)

    with gr.Row():
        aadhaar_no = gr.Textbox(label="Aadhaar Number (form, without space)", visible=True)
        address    = gr.Textbox(label="Address (form)", lines=2, visible=False)
        pan_no     = gr.Textbox(label="PAN Number (form)", visible=False)
        father     = gr.Textbox(label="Father's Name (form)", visible=False)

    with gr.Row():
        pan_name_form = gr.Textbox(label="PAN Name (form)", visible=False)

    run_btn = gr.Button("Verify & Generate Report", variant="primary")
    out_json = gr.JSON(label="Raw Output (extracted + scores)")
    out_html = gr.HTML(label="Human-readable Report")

    # Dynamically show/hide inputs based on selection (Change #4)
    def on_select(selected):
        sel = set(selected or [])

        # image visibility
        af_vis = "Aadhaar Front" in sel
        ab_vis = "Aadhaar Back" in sel
        pn_vis = "PAN" in sel

        # form fields visibility
        name_vis   = ("Aadhaar Front" in sel) or ("PAN" in sel)
        dob_vis    = ("Aadhaar Front" in sel) or ("PAN" in sel)
        gender_vis = ("Aadhaar Front" in sel)
        aadhaar_vis= ("Aadhaar Front" in sel)
        addr_vis   = ("Aadhaar Back" in sel)
        pan_vis_f  = ("PAN" in sel)
        father_vis = ("PAN" in sel)
        pan_name_vis = ("PAN" in sel)

        return (
            gr.update(visible=af_vis),
            gr.update(visible=ab_vis),
            gr.update(visible=pn_vis),
            gr.update(visible=name_vis),
            gr.update(visible=dob_vis),
            gr.update(visible=gender_vis),
            gr.update(visible=aadhaar_vis),
            gr.update(visible=addr_vis),
            gr.update(visible=pan_vis_f),
            gr.update(visible=father_vis),
            gr.update(visible=pan_name_vis),
        )

    doc_select.change(
        fn=on_select,
        inputs=[doc_select],
        outputs=[aadhaar_front_img, aadhaar_back_img, pan_img,
                 name, dob, gender, aadhaar_no, address, pan_no, father, pan_name_form]
    )

    def on_click(selected, af, ab, pn, nm, db, gd, an, addr, pno, fn, pn_name):
        sel = set(selected or [])
        # If user selected a doc but didn’t upload its image, we still allow proceeding,
        # but that doc’s extraction/verification will be skipped gracefully.
        return run_pipeline(
            list(sel),
            af if "Aadhaar Front" in sel else None,
            ab if "Aadhaar Back"  in sel else None,
            pn if "PAN"           in sel else None,
            nm or "", db or "", gd or "", an or "", addr or "", pno or "", pn_name or "", fn or ""
        )

    run_btn.click(
        fn=on_click,
        inputs=[doc_select, aadhaar_front_img, aadhaar_back_img, pan_img, name, dob, gender, aadhaar_no, address, pan_no, father, pan_name_form],
        outputs=[out_json, out_html]
    )

if __name__ == "__main__":
    # Make a public link if you want: set share=True
    demo.queue(api_open=False, max_size=2).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # set True to get a public URL
    )
