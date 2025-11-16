import re
import unicodedata
from datetime import datetime
from typing import Tuple


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)
    )


def _normalize_text(s: str) -> str:
    s = _strip_accents(s)
    s = s.replace("\u200b", "").replace("\xa0", " ")
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_simple(s: str) -> str:
    return _normalize_text(s).upper()


def _normalize_digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _normalize_date_str(s: str) -> str:
    s = (s or "").strip().replace("\\", "/").replace("-", "/").replace(".", "/")
    s = re.sub(r"\s+", "", s)

    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if m:
        d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y <= 30 else 1900
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


def _words(s: str) -> list[str]:
    return [w for w in _normalize_text(s).split(" ") if w]


def score_exact_string(form_value: str, extracted_value: str) -> Tuple[int, str]:
    """
    Exact equality (case/space-insensitive) => 100, else 0.
    Used for fields like gender that are textual but should match exactly.
    """
    f = _normalize_simple(form_value)
    e = _normalize_simple(extracted_value)
    if not f or not e:
        return 0, "Missing value"
    if f == e:
        return 100, "Exact match"
    return 0, f"Mismatch ('{form_value}' vs '{extracted_value}')"


def score_exact_digits(form_value: str, extracted_value: str) -> Tuple[int, str]:
    """
    Exact equality on digits only (Aadhaar, PAN) => 100, else 0.
    """
    # Explicitly remove spaces first, then keep digits only
    form_value = (form_value or "").replace(" ", "")
    extracted_value = (extracted_value or "").replace(" ", "")
    f = _normalize_digits(form_value)
    e = _normalize_digits(extracted_value)
    if not f or not e:
        return 0, "Missing value"
    if f == e:
        return 100, "Exact match"
    return 0, f"Mismatch ({f} vs {e})"


def score_exact_date(form_value: str, extracted_value: str) -> Tuple[int, str]:
    """
    Dates must match exactly after normalization (DD/MM/YYYY, YYYY/MM/DD, etc.).
    If normalized forms are equal => 100, otherwise 0.
    """
    if not form_value or not extracted_value:
        return 0, "Missing value"
    f = _normalize_date_str(form_value)
    e = _normalize_date_str(extracted_value)
    if f == e:
        return 100, f"Exact date match ({f})"
    return 0, f"Date mismatch ({f} vs {e})"


def score_proportional_words(form_value: str, extracted_value: str) -> Tuple[int, str]:
    """
    Proportional word match for name / father's name / address.

    - Split both sides into normalized word sets.
    - Let matches = |intersection|, total = |set1| + |set2|.
    - Score = ((matches / total) / 2) * 100  (as per your example).
      Example: 'Parth Patel' vs 'Parth Rajeshbhai Patel'
        set1 = {Parth, Patel}, set2 = {Parth, Rajeshbhai, Patel}
        matches = 2, total = 5
        score = ((2/5)/2)*100 = 20
    """
    w1 = set(_words(form_value.lower()))
    w2 = set(_words(extracted_value.lower()))
    if not w1 or not w2:
        return 0, "Missing value"

    matches = len(w1 & w2)
    total = len(w1) + len(w2)
    if total == 0:
        return 0, "Missing value"

    proportion = matches / total
    score = int(round((proportion*2.0) * 100))
    explanation = (
        f"{matches} of {total} unique words matched; "
        f"proportion={proportion:.2f}, score={score}"
    )
    return score, explanation
