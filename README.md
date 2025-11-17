st# KYC Document Consistency Checker

This project provides:

- A Gradio web UI to validate consistency between **form inputs** and information extracted from:
  - Aadhaar Front
  - Aadhaar Back
  - PAN Card
- Batch inference scripts to run OCR and export results for **Aadhaar** and **PAN** folders.

Underlying OCR is handled by `nanonets/Nanonets-OCR-s` via the Transformers vision-language model.

---

## 1. Environment & Dependencies

Install the main Python dependencies (adjust as needed for your environment):

```bash
pip install -r requirements.txt
```

The model `nanonets/Nanonets-OCR-s` is downloaded automatically by `transformers` on first use.

---

## 2. Gradio UI: Interactive Validation

Main entrypoint: `app_gradio_kyc.py`

### 2.1 Launch the app

From the `Project` directory:

```bash
python app_gradio_kyc.py
```

By default this:

- Starts Gradio on `http://0.0.0.0:7860`
- Enables a shareable public URL (`share=True`).

### 2.2 Supported documents

At the top of the UI, choose one or more documents to validate:

- `Aadhaar Front`
- `Aadhaar Back`
- `PAN`

The UI dynamically shows/hides:

- Image upload components for each selected document.
- Form fields required to validate those documents.

### 2.3 Form fields (typed values)

Depending on selection, you may see the following fields:

- `Full Name (form)`
- `Date of Birth (DD/MM/YYYY)`
- `Gender (form)` (dropdown: Male/Female/Other)
- `Aadhaar Number (form, without space)`
- `Address (form)`
- `PAN Number (form)`
- `PAN Name (form)`
- `Father's Name (form)`

> Note: Aadhaar number should be typed **without spaces**, but the validator also strips spaces before comparison.

### 2.4 What the model extracts

The OCR model is reused for all documents. Prompts:

- **Aadhaar Front** (`PROMPT_AADHAAR_FRONT`):
  - Extracts: `name`, `DOB`, `gender`, `aadhar_number`.
- **Aadhaar Back** (`PROMPT_AADHAAR_BACK`):
  - Extracts: `address`, `aadhar_number`.
- **PAN** (`PROMPT_PAN`):
  - Extracts: `permanent_account_number`, `name`, `father_s_name`, `date_of_birth`.

The raw JSON from the model is normalized using `canonicalize_keys(...)` in `app_gradio_kyc.py` to map to stable keys such as `name`, `DOB`, `gender`, `aadhar_number`, `address`, `permanent_account_number`, `father_s_name`, `date_of_birth`.

### 2.5 Scoring rules (0–100)

Scoring helpers live in `text_scoring.py` and are imported into `app_gradio_kyc.py`. Core rules:

#### Exact match fields (0 or 100)

- **Gender** (`score_exact_string`)
- **Aadhaar Number** (`score_exact_digits`)
- **PAN Number** (`score_exact_digits`)
- **DOB** (`score_exact_date`)

Details:

- `score_exact_string`:
  - Normalizes case/whitespace/punctuation.
  - 100 if equal, else 0.
- `score_exact_digits`:
  - Removes spaces, keeps digits only on both sides.
  - 100 if digit strings match, else 0.
- `score_exact_date`:
  - Normalizes various formats (e.g. `DD/MM/YYYY`, `YYYY/MM/DD`, `DDMMYYYY`).
  - 100 if normalized dates are equal, else 0.

#### Proportional word-match fields (0–100)

Used for:

- Name (form vs best of Aadhaar/PAN)
- PAN Name (form vs PAN)
- Father's Name (form vs PAN)
- Address (form vs Aadhaar Back)

Logic (`score_proportional_words`):

- Normalize strings and split into word sets: `w1`, `w2`.
- `matches = |w1 ∩ w2|`, `total = |w1| + |w2|`.
- Score: `((matches / total) / 2) * 100` (as per your custom design).

Example: `Parth Patel` vs `Parth Rajeshbhai Patel` ⇒

- `set1 = {PARTH, PATEL}`, `set2 = {PARTH, RAJESHBHAI, PATEL}`
- `matches = 2`, `total = 5`
- Score = `((2/5)*2)*100 = 80`.

#### Cross-document consistency rules

Additional hard checks:

- **Aadhaar number (Front vs Back)**:
  - If both front and back have Aadhaar numbers:
    - Normalize to digits (ignoring spaces).
    - If they differ, **Aadhaar score is forced to 0**, and explanations show both values.
  - If only one side has Aadhaar number:
    - Compare typed Aadhaar to that number with `score_exact_digits`.

- **DOB (Aadhaar vs PAN)**:
  - If both Aadhaar and PAN provide DOB:
    - Normalize via `normalize_date`.
    - If normalized values differ, **DOB score is forced to 0**, explanations show both.
  - Otherwise:
    - Compare typed DOB to whichever DOB(s) exist (Aadhaar and/or PAN) using `score_exact_date` and take the best score.

### 2.6 Overall score and severity

Per-field scores are combined with weights (`aggregate_score` in `app_gradio_kyc.py`):

- `name`: 0.25  
- `dob`: 0.20  
- `address`: 0.20  
- `pan_number`: 0.15  
- `aadhaar_number`: 0.15  
- `gender`: 0.05  

Overall score ∈ [0,100]. Severity labels:

- `LOW` (>= 90)
- `MEDIUM` (>= 70 and < 90)
- `HIGH` (< 70)

### 2.7 UI outputs

The UI returns two outputs:

1. **JSON** (`Raw Output (extracted + scores)`):
   - Contains:
     - `typed_form`: typed values.
     - `extracted`: normalized OCR from each document.
     - `scores`: per-field scores and explanations.
     - `overall_score`, `severity`.
2. **HTML** (`Human-readable Report`):
   - A table with columns:
     - Field
     - Typed (Form)
     - Extracted from Document(s)
     - Score
     - Explanation
   - Shows Aadhaar front/back values where applicable (e.g. Aadhaar number row).

---

## 3. Batch Inference

Batch scripts let you run OCR on folders of images without the UI.

### 3.1 Unified batch script: `batch_infer_all.py`

This is the recommended entrypoint for batch runs. It supports three subcommands:

#### 3.1.1 Aadhaar front

```bash
python batch_infer_all.py aadhaar_front "./data/Adhar Front" -o "./result/final_aadhaar_front.xlsx" -b 4
```

- Input: folder of Aadhaar **front** images.
- Output: Excel (`aadhaar_front.xlsx` by default) with columns:
  - `file`
  - `name`
  - `gender`
  - `dob`
  - `aadhaar_number` (digits-only; spaces and non-digits removed via `clean_aadhaar_number`)

#### 3.1.2 Aadhaar back

```bash
python batch_infer_all.py aadhaar_back "./data/Adhar Back" -o "./result/final_aadhaar_back.xlsx" -b 4
```

- Input: folder of Aadhaar **back** images.
- Output: XLSX (`aadhaar_back.xlsx` by default) with columns:
  - `file`
  - `aadhaar_number`
  - `address`

#### 3.1.3 PAN

```bash
python batch_infer_all.py pan "./data/Pan" -o "./result/final_pan.xlsx" -b 4
```

- Input: folder of **PAN** images.
- Output: Excel (`pan.xlsx` by default) with columns:
  - `file`
  - `permanent_account_number`
  - `name`
  - `father_s_name`
  - `date_of_birth`
- Keys are normalized using `canonicalize_keys` from `app_gradio_kyc.py`.



---

## 4. Typical Workflow

1. **Collect documents**:
   - Aadhaar front, Aadhaar back, and PAN images from the customer.
2. **Run OCR batch (optional)**:
   - Use `batch_infer_all.py` to get CSV/XLSX exports for bulk datasets.
3. **Interactive validation**:
   - For a single customer/session, launch `app_gradio_kyc.py`.
   - Upload the selected documents, fill in the form, and click **Verify & Generate Report**.
4. **Review results**:
   - Check per-field scores and explanations in the HTML report.
   - Pay special attention to: 
     - Aadhaar number front vs back.
     - DOB Aadhaar vs PAN.
     - Name / PAN Name / Father’s Name / Address proportional matches.

This setup gives you both a **human-friendly interactive tool** and **automation-friendly batch tools** using the same OCR backbone and normalization logic.

