"""
Legal Named Entity Recognition (NER) and Judicial Outcome Extraction for Indian Judgments.
Extracts:
- Statutes & Central Acts
- Sections & Articles
- Courts & Judicial Authorities
- Judges & Bench Strength (Single, Division, Full, Constitution Bench)
- Landmark Case Citations (SCC, AIR, SCR)
- Final Verdict / Case Disposition (Allowed, Dismissed, Remanded, Disposed)
"""

import re
from typing import Dict, List, Any, Set


# Common Indian Central Acts
INDIAN_STATUTES = [
    r"Constitution of India",
    r"Indian Penal Code(?:,?\s*1860)?",
    r"Code of Criminal Procedure(?:,?\s*1973)?",
    r"Code of Civil Procedure(?:,?\s*1908)?",
    r"Indian Evidence Act(?:,?\s*1872)?",
    r"Indian Contract Act(?:,?\s*1872)?",
    r"Negotiable Instruments Act(?:,?\s*1881)?",
    r"Specific Relief Act(?:,?\s*1963)?",
    r"Limitation Act(?:,?\s*1963)?",
    r"Arbitration and Conciliation Act(?:,?\s*1996)?",
    r"Information Technology Act(?:,?\s*2000)?",
    r"Companies Act(?:,?\s*(?:1956|2013))?",
    r"Narcotic Drugs and Psychotropic Substances Act(?:,?\s*1985)?|\bNDPS Act\b",
    r"Protection of Children from Sexual Offences Act(?:,?\s*2012)?|\bPOCSO Act\b",
    r"Prevention of Corruption Act(?:,?\s*1988)?",
    r"Prevention of Money Laundering Act(?:,?\s*2002)?|\bPMLA\b",
    r"Insolvency and Bankruptcy Code(?:,?\s*2016)?|\bIBC\b",
    r"Motor Vehicles Act(?:,?\s*1988)?",
    r"Consumer Protection Act(?:,?\s*(?:1986|2019))?",
    r"Right to Information Act(?:,?\s*2005)?|\bRTI Act\b",
    r"[A-Z][A-Za-z\s]+ Act,?\s+\d{4}"
]

# Indian Courts
INDIAN_COURTS = [
    r"Supreme Court of India",
    r"(?:Bombay|Delhi|Madras|Calcutta|Allahabad|Karnataka|Kerala|Punjab and Haryana|Gujarat|Rajasthan|Patna|Madhya Pradesh|Orissa|Gauhati|Jammu and Kashmir|Andhra Pradesh|Telangana|Himachal Pradesh|Jharkhand|Uttarakhand|Chhattisgarh) High Court",
    r"High Court of (?:Bombay|Delhi|Madras|Calcutta|Allahabad|Karnataka|Kerala|Punjab and Haryana|Gujarat|Rajasthan|Patna|Madhya Pradesh|Orissa|Gauhati|Judicature at [A-Za-z]+)",
    r"National Company Law Appellate Tribunal|\bNCLAT\b",
    r"National Company Law Tribunal|\bNCLT\b",
    r"National Green Tribunal|\bNGT\b",
    r"Central Administrative Tribunal|\bCAT\b",
    r"District and Sessions Court|\bSessions Court\b|\bDistrict Court\b|\bTrial Court\b|\bMagistrate Court\b"
]

# Statutory Sections & Articles
SECTION_PATTERNS = [
    r"\b(?:Section|Sec\.|Sections|Secs\.)\s+\d+[A-Z]?(?:\s*\(\d+\))*(?:\s*(?:read with|r/w)\s*(?:Section|Sec\.)?\s*\d+[A-Z]?)?",
    r"\b(?:Article|Art\.|Articles|Arts\.)\s+\d+[A-Z]?(?:\s*\(\d+\))*(?:\([a-z]\))*",
    r"\bOrder\s+[A-Z0-9]+\s+Rule\s+\d+[A-Z]?"
]

# Citation Formats (SCC, AIR, SCR, MANU)
CITATION_PATTERNS = [
    r"\(\d{4}\)\s*\d+\s*SCC\s*\d+",                    # (2018) 10 SCC 1
    r"\bAIR\s+\d{4}\s+(?:SC|Bom|Del|Cal|Mad|All)\s+\d+", # AIR 1973 SC 1461
    r"\[\d{4}\]\s*\d+\s*SCR\s*\d+",                    # [1973] 4 SCR 1
    r"\bMANU\/[A-Z]{2}\/\d{4}\/\d{4}\b",               # MANU/SC/0123/2020
    r"\d{4}\s+Cri\s*LJ\s*\d+"                          # 2021 Cri LJ 500
]


def extract_all_entities(text: str) -> Dict[str, List[str]]:
    """
    Extracts structured legal entities from the judgment text.
    """
    entities: Dict[str, Set[str]] = {
        "Statutes & Acts": set(),
        "Sections & Articles": set(),
        "Courts": set(),
        "Case Citations": set(),
        "Judges": set(),
        "Dates": set()
    }

    # 1. Statutes & Acts
    for pattern in INDIAN_STATUTES:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            cleaned = re.sub(r"\s+", " ", m).strip()
            if len(cleaned) > 4:
                entities["Statutes & Acts"].add(cleaned)

    # 2. Courts
    for pattern in INDIAN_COURTS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            cleaned = re.sub(r"\s+", " ", m).strip()
            entities["Courts"].add(cleaned)

    # 3. Sections & Articles
    for pattern in SECTION_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            cleaned = re.sub(r"\s+", " ", m).strip()
            entities["Sections & Articles"].add(cleaned)

    # 4. Citations
    for pattern in CITATION_PATTERNS:
        matches = re.findall(pattern, text)
        for m in matches:
            entities["Case Citations"].add(m.strip())

    # 5. Judges
    judge_matches = re.findall(
        r"(?:Justice|Hon'ble Mr\. Justice|Hon'ble Mrs\. Justice|Hon'ble Ms\. Justice)\s+([A-Z][A-Za-z\s.]+)|"
        r"([A-Z][A-Za-z\s.]+),\s*(?:J\.|C\.J\.|CJI)",
        text
    )
    for match in judge_matches:
        j_name = match[0] or match[1]
        j_name = re.sub(r"\s+", " ", j_name).strip()
        # Filter noise
        if 4 < len(j_name) < 40 and not any(kw in j_name.lower() for kw in ["court", "order", "state", "appellant"]):
            entities["Judges"].add(j_name)

    # 6. Dates
    date_matches = re.findall(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|"
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|"
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b",
        text,
        re.IGNORECASE
    )
    for d in date_matches:
        entities["Dates"].add(d.strip())

    return {
        key: sorted(list(val))[:20] for key, val in entities.items()
    }


def determine_bench_type(judges: List[str]) -> Dict[str, str]:
    """
    Identifies the bench strength in Indian judicial nomenclature.
    """
    count = len(judges)
    if count >= 5:
        return {
            "type": "Constitution Bench",
            "desc": f"{count}-Judge Constitution Bench (deciding substantial questions of constitutional interpretation)",
            "count": str(count)
        }
    elif count == 3:
        return {
            "type": "Full Bench",
            "desc": "3-Judge Full Bench",
            "count": "3"
        }
    elif count == 2:
        return {
            "type": "Division Bench",
            "desc": "2-Judge Division Bench",
            "count": "2"
        }
    elif count == 1:
        return {
            "type": "Single Judge Bench",
            "desc": "Single Judge Bench",
            "count": "1"
        }
    else:
        return {
            "type": "Appellate Bench",
            "desc": "Appellate Bench",
            "count": "Unspecified"
        }


def extract_verdict_outcome(text: str) -> Dict[str, str]:
    """
    Detects the final legal outcome and disposition of the judgment.
    """
    last_portion = text[-4000:] if len(text) > 4000 else text

    # Appeal Allowed / Acquittal
    if re.search(
        r"(?:appeal\s+(?:is\s+)?allowed|"
        r"judgment\s+(?:and\s+order\s+)?of\s+conviction\s+is\s+set\s+aside|"
        r"appellant\s+is\s+acquitted|"
        r"impugned\s+order\s+is\s+(?:quashed|set\s+aside)|"
        r"petition\s+is\s+allowed|"
        r"bail\s+is\s+granted)",
        last_portion,
        re.IGNORECASE
    ):
        return {
            "status": "ALLOWED / RELIEF GRANTED",
            "badge_color": "green",
            "summary": "The appeal/petition was ALLOWED. The impugned lower court order was set aside or the conviction was overturned."
        }

    # Appeal Dismissed / Conviction Upheld
    elif re.search(
        r"(?:appeal\s+(?:is\s+)?dismissed|"
        r"find\s+no\s+merit\s+in\s+the\s+appeal|"
        r"conviction\s+is\s+(?:confirmed|affirmed|upheld)|"
        r"petition\s+is\s+dismissed|"
        r"order\s+of\s+the\s+High\s+Court\s+is\s+upheld|"
        r"no\s+ground\s+for\s+interference)",
        last_portion,
        re.IGNORECASE
    ):
        return {
            "status": "DISMISSED / CONVICTION UPHELD",
            "badge_color": "red",
            "summary": "The appeal/petition was DISMISSED. The higher court found no error in the lower court's decision."
        }

    # Remanded / Sent back
    elif re.search(
        r"(?:remanded\s+back|remitted\s+to\s+the\s+(?:High|trial)\s+Court|re-trial|fresh\s+consideration)",
        last_portion,
        re.IGNORECASE
    ):
        return {
            "status": "REMANDED FOR RECONSIDERATION",
            "badge_color": "orange",
            "summary": "The case was REMANDED back to the lower court for fresh examination in light of legal principles."
        }

    # Disposed with directions / Partly allowed
    elif re.search(
        r"(?:partly\s+allowed|disposed\s+of\s+with\s+(?:the\s+following\s+)?directions|modified\s+accordingly)",
        last_portion,
        re.IGNORECASE
    ):
        return {
            "status": "PARTLY ALLOWED / DISPOSED WITH DIRECTIONS",
            "badge_color": "blue",
            "summary": "The appeal was PARTLY ALLOWED or DISPOSED with specific judicial directions."
        }

    else:
        return {
            "status": "DISPOSED / ADJUDICATED",
            "badge_color": "gray",
            "summary": "The proceedings were concluded in accordance with the court's reasoning and observations."
        }
