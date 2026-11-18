"""
Text Preprocessing and Legal Sentence Tokenization.
Designed specifically for Indian Court Judgments and Legal Documents.
"""

import re
from typing import List, Dict, Any, Optional

# Common Indian legal abbreviations that should NOT trigger a sentence split
LEGAL_ABBREVIATIONS = (
    r"(?:AIR|SCC|SCR|CrLJ|ILR|MANU|ILJ|FLR|"
    r"Dr|Mr|Mrs|Ms|Hon'ble|J|JJ|CJ|CJI|"
    r"v|vs|v/s|No|Nos|Sec|Secs|Art|Arts|para|paras|cl|cls|"
    r"Ltd|Pvt|Govt|Ors|Anr|Co|Inc|Corp|viz|i\.e|e\.g|etc|cf|ibid|id)\."
)


def legal_sent_tokenize(text: str) -> List[str]:
    """
    Splits legal judgment text into clean, coherent sentences while strictly
    preserving legal citations, statutory abbreviations, judge initials, and dates.
    Falls back gracefully without external dependencies (like NLTK punkt_tab).
    """
    if not text or not isinstance(text, str):
        return []

    # Standardize line endings
    normalized = re.sub(r"\r\n|\r", "\n", text)

    # Temporary mask for abbreviations
    def mask_match(m):
        return m.group(0).replace(".", "<DOT>")

    # 1. Mask periods in legal abbreviations
    masked = re.sub(LEGAL_ABBREVIATIONS, mask_match, normalized, flags=re.IGNORECASE)

    # 2. Mask periods in dates (e.g., 14.08.2021) and statutory section decimals (e.g., 304.B)
    masked = re.sub(r"(\d+)\.(\d+)", r"\1<DOT>\2", masked)

    # 3. Mask initials like D.Y. or K.S.
    masked = re.sub(r"\b([A-Z])\.\s*([A-Z])\.", r"\1<DOT> \2<DOT>", masked)
    masked = re.sub(r"\b([A-Z])\.", r"\1<DOT>", masked)

    # 4. Mask citations like (2018) 10 SCC 1.
    masked = re.sub(r"(\(\d{4}\)\s*\d+\s*[A-Z]+)\.", r"\1<DOT>", masked)

    # 5. Split sentences on terminal punctuation (. ! ?) followed by whitespace and a capital letter/quote/number
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", masked)

    clean_sentences = []
    for s in raw_sentences:
        # Unmask dots
        s = s.replace("<DOT>", ".")
        # Remove paragraph numbers like "1. ", "23. ", "[14] ", "(1) "
        s = re.sub(r"^(?:\[\d+\]|\(\d+\)|\d+\.)\s*", "", s.strip())
        # Collapse multi-spaces
        s = re.sub(r"\s+", " ", s).strip()
        # Filter out noisy or overly short fragments (less than 15 chars)
        if len(s) >= 15:
            clean_sentences.append(s)

    return clean_sentences


def clean_legal_text(text: str) -> str:
    """Normalizes whitespace and standardizes quote characters."""
    if not text:
        return ""
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_document_header(text: str) -> Dict[str, Optional[str]]:
    """
    Extracts high-level Indian judgment metadata:
    - Court Name
    - Appeal / Petition Number
    - Appellant / Petitioner
    - Respondent
    - Judges / Bench
    """
    metadata: Dict[str, Optional[str]] = {
        "court": None,
        "case_no": None,
        "petitioner": None,
        "respondent": None,
        "bench": None,
    }

    # Court
    court_match = re.search(
        r"(IN THE SUPREME COURT OF INDIA|"
        r"IN THE HIGH COURT OF [A-Z\s]+|"
        r"SUPREME COURT OF INDIA|"
        r"[A-Z\s]+ HIGH COURT)",
        text,
        re.IGNORECASE,
    )
    if court_match:
        metadata["court"] = court_match.group(0).strip().title()

    # Case Number
    case_match = re.search(
        r"(CRIMINAL|CIVIL|SPECIAL LEAVE|WRIT)\s+(?:APPEAL|PETITION)\s+(?:NO\.|NUMBER)?\s*[\d\s/OFof-]+",
        text,
        re.IGNORECASE,
    )
    if case_match:
        metadata["case_no"] = re.sub(r"\s+", " ", case_match.group(0)).strip().upper()

    # Parties (Petitioner vs Respondent)
    parties_match = re.search(
        r"([A-Z0-9\s.,'&()-]+?)\s+(?:\.\.\.|\s+)(?:APPELLANT|PETITIONER)\(?S?\)?\s*(?:VERSUS|VS\.?|V/S)\s*([A-Z0-9\s.,'&()-]+?)\s+(?:\.\.\.|\s+)(?:RESPONDENT|DEFENDANT)",
        text,
        re.IGNORECASE,
    )
    if parties_match:
        pet = re.sub(r"\s+", " ", parties_match.group(1)).strip().title()
        resp = re.sub(r"\s+", " ", parties_match.group(2)).strip().title()
        metadata["petitioner"] = pet[:80]
        metadata["respondent"] = resp[:80]

    # Bench / Judge
    judge_match = re.search(
        r"(?:BEFORE|CORAM|JUDGMENT BY|HON'BLE)\s*[:\-]?\s*([A-Z\s.,()]+(?:,\s*J\.?|,\s*C\.?J\.?|JUSTICE\s+[A-Z\s.]+))",
        text,
        re.IGNORECASE,
    )
    if judge_match:
        metadata["bench"] = re.sub(r"\s+", " ", judge_match.group(1)).strip()

    return metadata


def extract_text_from_upload(uploaded_file) -> str:
    """Safely extracts text from uploaded .txt or .pdf files."""
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        try:
            return uploaded_file.read().decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error reading text file: {e}"

    elif filename.endswith(".pdf"):
        # Try PyMuPDF (fitz)
        try:
            import fitz

            pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            pages = [page.get_text() for page in pdf]
            pdf.close()
            return "\n".join(pages)
        except ImportError:
            pass
        except Exception as e:
            return f"Error reading PDF with PyMuPDF: {e}"

        # Try pypdf
        try:
            import pypdf

            reader = pypdf.PdfReader(uploaded_file)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except ImportError:
            pass
        except Exception as e:
            return f"Error reading PDF with pypdf: {e}"

        return (
            "PDF extraction requires 'pymupdf' or 'pypdf'. "
            "Please paste the judgment text directly into the raw text input or upload a .txt file."
        )

    return ""
