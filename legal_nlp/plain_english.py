"""
Plain-English Simplification Engine for Indian Legal Texts.
Provides:
1. Indian Legal Jargon & Latin Maxim Explainer.
2. Statutory Section & Article Meaning Lookup.
3. Quantitative Readability Metrics (Flesch-Kincaid, Gunning Fog).
4. Automated Layman Case Brief Generation.
"""

import re
import math
from typing import Dict, List, Any, Tuple


# Comprehensive Dictionary of Latin Maxims and Indian Legal Terminology
LEGAL_JARGON_GLOSSARY = {
    # Latin Maxims
    "prima facie": "On first impression / at first glance; accepted as correct until proven otherwise.",
    "res judicata": "A matter already definitively judged by a competent court cannot be re-litigated by the same parties.",
    "mens rea": "Guilty mind / criminal intention, an essential mental element of a criminal offence.",
    "actus reus": "The wrongful physical act or illegal omission constituting a crime.",
    "ratio decidendi": "The core legal principle or rationale on which the court's binding judgment is based.",
    "obiter dicta": "Incidental remarks or observations made in passing by a judge, which are not legally binding.",
    "suo motu": "On its own motion; action initiated by the court itself without a petition from any party.",
    "inter alia": "Among other things / including others.",
    "locus standi": "The legal right or qualification of an individual to bring a case or appear before a court.",
    "ultra vires": "Beyond legal power or constitutional authority; therefore invalid and void.",
    "intra vires": "Within legal or constitutional power and authority.",
    "habeas corpus": "A constitutional court writ directing the release of a person held in unlawful detention.",
    "mandamus": "A court order compelling a public authority or official to execute a mandatory statutory duty.",
    "certiorari": "A higher court's writ quashing an illegal or jurisdiction-exceeding order of a lower court.",
    "quo warranto": "A writ questioning by what legal authority a person holds a public office.",
    "bona fide": "In good faith; genuine, sincere, and without deceptive intent.",
    "mala fide": "In bad faith; malicious, dishonest, or carried out with improper motives.",
    "in limine": "At the threshold or preliminary stage; dismissed at the very outset.",
    "ex parte": "Proceedings conducted or orders issued for one party without hearing the opponent.",
    "sub judice": "Currently under judicial consideration and therefore not open to public interference.",
    "status quo": "The existing state of affairs; an order maintaining things as they currently stand.",
    "de facto": "In reality or in practice, whether officially recognized by law or not.",
    "de jure": "According to lawful right, official legality, or recognized law.",
    "mutatis mutandis": "With the necessary changes made in points of detail; adapted accordingly.",
    "corpus delicti": "The objective physical body of evidence that proves a crime was actually committed.",
    "sine qua non": "An indispensable, essential condition without which something cannot happen.",
    "res ipsa loquitur": "The thing speaks for itself; negligence is inferred from the obvious nature of the accident.",
    "amicus curiae": "'Friend of the court'; an independent advocate invited to assist the bench on complex law.",
    "lis pendens": "Pending legal action; a rule prohibiting the transfer of contested property during a lawsuit.",
    "in camera": "A court hearing held in private or behind closed doors, without press or public present.",
    "nolle prosequi": "Formal notice that the prosecutor is abandoning or dropping a criminal prosecution.",

    # Indian Legal Terminology & Procedural Words
    "impugned order": "The lower court or tribunal decision that is being challenged or appealed against.",
    "special leave petition": "An appeal petition filed under Article 136 before the Supreme Court seeking special leave to appeal.",
    "slp": "Special Leave Petition under Article 136 of the Constitution of India.",
    "quash": "To officially void, invalidate, and erase a criminal FIR, charge-sheet, or lower court order.",
    "cognizable": "A serious criminal offence where police can investigate and arrest without an arrest warrant.",
    "non-cognizable": "A minor offence where police cannot arrest without a magistrate's prior warrant.",
    "non-bailable": "An offence where bail is not an absolute right and lies solely within the judge's discretion.",
    "anticipatory bail": "Pre-arrest protection granted under Section 438 CrPC against anticipated detention.",
    "remand": "Sending an arrested suspect back into custody (police or judicial) or sending a case back to trial court.",
    "acquittal": "A formal judgment declaring that the accused has been cleared of all criminal charges.",
    "conviction": "A formal court ruling that the accused is guilty of the charged offences.",
    "stay order": "A temporary judicial injunction halting the execution or enforcement of a decision.",
    "caveat": "A preventive legal notice requesting the court not to pass any adverse order without prior hearing.",
    "injunction": "A court order requiring a person to do or refrain from doing a specific act.",
    "appellant": "The party appealing a decision to a higher appellate court.",
    "respondent": "The opposing party responding to or defending against an appeal.",
    "petitioner": "The person or entity who files a formal petition (such as a writ petition) in court.",
    "plaintiff": "The person who initiates a civil lawsuit seeking legal remedies.",
    "defendant": "The person or entity defending against a civil lawsuit.",
    "decree": "The formal expression of an adjudication conclusively determining the rights of civil parties.",
    "prosecution": "The government / legal team conducting criminal proceedings against an accused on behalf of the state."
}

# Statutory Sections Lookup (IPC, CrPC, Constitution)
STATUTE_KNOWLEDGE_BASE = {
    # IPC
    "Section 302 IPC": "Punishment for Murder (mandatory death penalty or imprisonment for life, and liable to fine).",
    "Section 304 IPC": "Culpable Homicide not amounting to murder (punishable with imprisonment for life or up to 10 years).",
    "Section 304B IPC": "Dowry Death (death of a woman caused by burns/bodily injury within 7 years of marriage under cruelty for dowry).",
    "Section 307 IPC": "Attempt to Murder (punishable with imprisonment up to 10 years or life if hurt is caused).",
    "Section 376 IPC": "Punishment for Rape (rigorous imprisonment not less than 10 years, up to life imprisonment).",
    "Section 420 IPC": "Cheating and dishonestly inducing delivery of property (punishable with imprisonment up to 7 years + fine).",
    "Section 406 IPC": "Punishment for Criminal Breach of Trust (imprisonment up to 3 years or fine).",
    "Section 498A IPC": "Husband or relative of husband of a woman subjecting her to cruelty (imprisonment up to 3 years + fine).",
    "Section 120B IPC": "Punishment of Criminal Conspiracy (conspiring to commit an illegal act).",
    "Section 34 IPC": "Acts done by several persons in furtherance of common intention (joint liability principle).",
    "Section 149 IPC": "Every member of unlawful assembly guilty of offence committed in prosecution of common object.",
    "Section 201 IPC": "Causing disappearance of evidence of offence, or giving false information to screen offender.",

    # CrPC
    "Section 482 CrPC": "Inherent powers of the High Court to prevent abuse of legal process and secure the ends of justice (used to quash FIRs).",
    "Section 438 CrPC": "Direction for grant of Anticipatory Bail to person apprehending arrest.",
    "Section 439 CrPC": "Special powers of High Court or Court of Session regarding granting regular bail.",
    "Section 154 CrPC": "Information in cognizable cases (Registration of First Information Report / FIR).",
    "Section 161 CrPC": "Examination of witnesses by police during criminal investigation.",
    "Section 164 CrPC": "Recording of confessions and statements before a Judicial Magistrate.",
    "Section 313 CrPC": "Power of trial court to examine the accused personally after prosecution evidence.",
    "Section 173 CrPC": "Report of police officer on completion of investigation (Charge-sheet / Final Report).",

    # Constitution of India
    "Article 14": "Equality before law and equal protection of the laws within the territory of India.",
    "Article 19": "Protection of certain Fundamental Rights: freedom of speech, assembly, association, movement, and profession.",
    "Article 21": "Protection of Life and Personal Liberty: No person shall be deprived of life or personal liberty except by procedure established by law.",
    "Article 32": "Right to Constitutional Remedies: Direct right to move the Supreme Court by appropriate proceedings for enforcement of Fundamental Rights.",
    "Article 226": "Power of High Courts to issue writs (Habeas Corpus, Mandamus, Certiorari, Prohibition, Quo Warranto) for enforcement of rights.",
    "Article 136": "Special Leave to Appeal granted by Supreme Court against any judgment or order in any cause or matter.",
    "Article 141": "Law declared by Supreme Court shall be binding on all courts within the territory of India.",
    "Article 142": "Enforcement of decrees and orders of Supreme Court to do complete justice in any cause or matter.",

    # Other Acts
    "Section 138 NI Act": "Dishonour of cheque for insufficiency of funds in the bank account (criminal liability up to 2 years imprisonment).",
    "Section 73 Contract Act": "Compensation for loss or damage caused by breach of contract."
}


def detect_and_explain_jargon(text: str) -> List[Dict[str, Any]]:
    """
    Finds Latin maxims and technical Indian legal terms in the text
    and returns their occurrence counts, positions, and layman explanations.
    """
    detected = []
    text_lower = text.lower()

    for term, explanation in LEGAL_JARGON_GLOSSARY.items():
        pattern = r"\b" + re.escape(term) + r"\b"
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            detected.append({
                "term": term.title(),
                "occurrences": len(matches),
                "plain_meaning": explanation,
                "category": "Latin Maxim" if term in [
                    "prima facie", "res judicata", "mens rea", "actus reus",
                    "ratio decidendi", "obiter dicta", "suo motu", "inter alia",
                    "locus standi", "ultra vires", "bona fide", "mala fide",
                    "habeas corpus", "mandamus", "certiorari", "amicus curiae"
                ] else "Legal Terminology"
            })

    # Sort by frequency of appearance
    detected.sort(key=lambda x: x["occurrences"], reverse=True)
    return detected


def lookup_statutory_sections(text: str) -> List[Dict[str, str]]:
    """
    Intelligently scans judgment text for statutory sections and articles
    and returns their layman explanations.
    """
    results = []
    seen = set()
    text_lower = text.lower()

    is_ipc = "indian penal code" in text_lower or "ipc" in text_lower
    is_crpc = "code of criminal procedure" in text_lower or "crpc" in text_lower or "cr.p.c" in text_lower
    is_constitution = "constitution of india" in text_lower or "constitution" in text_lower

    # 1. Search key sections
    section_checks = [
        # IPC
        ("302", is_ipc, "Section 302 IPC", STATUTE_KNOWLEDGE_BASE.get("Section 302 IPC")),
        ("304", is_ipc, "Section 304 IPC", STATUTE_KNOWLEDGE_BASE.get("Section 304 IPC")),
        ("304b", is_ipc, "Section 304B IPC", STATUTE_KNOWLEDGE_BASE.get("Section 304B IPC")),
        ("307", is_ipc, "Section 307 IPC", STATUTE_KNOWLEDGE_BASE.get("Section 307 IPC")),
        ("376", is_ipc, "Section 376 IPC", STATUTE_KNOWLEDGE_BASE.get("Section 376 IPC")),
        ("420", is_ipc, "Section 420 IPC", STATUTE_KNOWLEDGE_BASE.get("Section 420 IPC")),
        ("498a", is_ipc, "Section 498A IPC", STATUTE_KNOWLEDGE_BASE.get("Section 498A IPC")),
        ("120b", is_ipc, "Section 120B IPC", STATUTE_KNOWLEDGE_BASE.get("Section 120B IPC")),
        ("34", is_ipc, "Section 34 IPC", STATUTE_KNOWLEDGE_BASE.get("Section 34 IPC")),
        # CrPC
        ("482", is_crpc, "Section 482 CrPC", STATUTE_KNOWLEDGE_BASE.get("Section 482 CrPC")),
        ("438", is_crpc, "Section 438 CrPC", STATUTE_KNOWLEDGE_BASE.get("Section 438 CrPC")),
        ("439", is_crpc, "Section 439 CrPC", STATUTE_KNOWLEDGE_BASE.get("Section 439 CrPC")),
        ("154", is_crpc, "Section 154 CrPC", STATUTE_KNOWLEDGE_BASE.get("Section 154 CrPC")),
        ("313", is_crpc, "Section 313 CrPC", STATUTE_KNOWLEDGE_BASE.get("Section 313 CrPC")),
        # Other
        ("138", "negotiable" in text_lower or "cheque" in text_lower or "ni act" in text_lower, "Section 138 NI Act", STATUTE_KNOWLEDGE_BASE.get("Section 138 NI Act")),
        ("73", "contract" in text_lower, "Section 73 Contract Act", STATUTE_KNOWLEDGE_BASE.get("Section 73 Contract Act")),
        ("74", "contract" in text_lower or "earnest" in text_lower, "Section 74 Contract Act", "Compensation for breach of contract where penalty or liquidated damages are stipulated.")
    ]

    for sec_num, condition, label, meaning in section_checks:
        pattern = r"\b(?:section|sec\.)\s+" + re.escape(sec_num) + r"\b"
        if re.search(pattern, text_lower) and condition and label not in seen and meaning:
            results.append({"citation": label, "meaning": meaning})
            seen.add(label)

    # 2. Search Articles of Constitution
    article_checks = [
        ("14", "Article 14", STATUTE_KNOWLEDGE_BASE.get("Article 14")),
        ("19", "Article 19", STATUTE_KNOWLEDGE_BASE.get("Article 19")),
        ("21", "Article 21", STATUTE_KNOWLEDGE_BASE.get("Article 21")),
        ("32", "Article 32", STATUTE_KNOWLEDGE_BASE.get("Article 32")),
        ("226", "Article 226", STATUTE_KNOWLEDGE_BASE.get("Article 226")),
        ("136", "Article 136", STATUTE_KNOWLEDGE_BASE.get("Article 136")),
        ("141", "Article 141", STATUTE_KNOWLEDGE_BASE.get("Article 141")),
        ("142", "Article 142", STATUTE_KNOWLEDGE_BASE.get("Article 142"))
    ]

    for art_num, label, meaning in article_checks:
        pattern = r"\b(?:article|art\.)\s+" + re.escape(art_num) + r"\b"
        if re.search(pattern, text_lower) and is_constitution and label not in seen and meaning:
            results.append({"citation": label, "meaning": meaning})
            seen.add(label)

    return results


def count_syllables(word: str) -> int:
    """Estimates the number of syllables in an English word."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]|ed|es|e)$", "", word)
    word = re.sub(r"^y", "", word)
    syllables = len(re.findall(r"[aeiouy]{1,2}", word))
    return max(1, syllables)


def calculate_readability(text: str) -> Dict[str, Any]:
    """
    Calculates readability metrics:
    - Flesch Reading Ease (0-100 scale: 0-30 = Very Difficult, 60-70 = Standard, 90+ = Very Easy)
    - Flesch-Kincaid Grade Level (US school grade level: 12 = High School, 16 = College, 20+ = Post-grad/Legal)
    - Gunning Fog Index
    - Average Sentence Length (Words per sentence)
    - Percentage of Complex Words (words with 3+ syllables)
    """
    words = re.findall(r"\b[A-Za-z]+\b", text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 5]

    total_words = max(1, len(words))
    total_sentences = max(1, len(sentences))

    complex_words = 0
    total_syllables = 0

    for w in words:
        syl = count_syllables(w)
        total_syllables += syl
        if syl >= 3:
            complex_words += 1

    asl = total_words / total_sentences  # Average Sentence Length
    asw = total_syllables / total_words  # Average Syllables per Word
    pct_complex = (complex_words / total_words) * 100

    # Flesch Reading Ease formula
    fre = 206.835 - (1.015 * asl) - (84.6 * asw)
    fre = max(0.0, min(100.0, round(fre, 1)))

    # Flesch-Kincaid Grade Level formula
    fkgl = (0.39 * asl) + (11.8 * asw) - 15.59
    fkgl = max(1.0, round(fkgl, 1))

    # Gunning Fog formula
    fog = 0.4 * (asl + pct_complex)
    fog = max(1.0, round(fog, 1))

    # Determine grade level interpretation
    if fkgl >= 18:
        level_label = "Extremely Dense Legal Text (Doctorate / Senior Judicial Level)"
        badge_color = "red"
    elif fkgl >= 15:
        level_label = "Difficult Legal Language (Law Graduate / Professional Level)"
        badge_color = "orange"
    elif fkgl >= 11:
        level_label = "Moderate Complexity (College / Educated Layman Level)"
        badge_color = "yellow"
    else:
        level_label = "Simplified Plain English (High School / General Public Level)"
        badge_color = "green"

    return {
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fkgl,
        "gunning_fog": fog,
        "avg_sentence_length": round(asl, 1),
        "pct_complex_words": round(pct_complex, 1),
        "total_words": total_words,
        "total_sentences": total_sentences,
        "level_label": level_label,
        "badge_color": badge_color,
    }


def generate_layman_brief(
    metadata: Dict[str, Any],
    role_sentences: Dict[str, List[str]],
    outcome_summary: str,
    top_statutes: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Synthesizes a structured 5-point Case Brief in Plain English:
    1. Background & Facts (What happened)
    2. Core Dispute (The key legal question)
    3. Petitioner's Argument
    4. Respondent's Defense
    5. Final Outcome & Why (Decision + Ratio)
    """
    facts = role_sentences.get("Facts", [])
    issues = role_sentences.get("Issue", [])
    pet_args = role_sentences.get("Arguments of Petitioner", [])
    resp_args = role_sentences.get("Arguments of Respondent", [])
    reasoning = role_sentences.get("Reasoning", [])
    decision = role_sentences.get("Decision", [])

    # Simplify Facts
    fact_text = (
        " ".join(facts[:3])
        if facts
        else "The dispute stems from lower court proceedings between the involved parties."
    )

    # Simplify Issue
    if issues:
        issue_text = issues[0]
    else:
        issue_text = "Whether the decision of the lower court was legally sustainable and in accordance with statutory evidence."

    # Arguments
    pet_text = (
        " ".join(pet_args[:2])
        if pet_args
        else "The petitioner contended that the lower court erred in evaluating evidence and violated procedural fairness."
    )
    resp_text = (
        " ".join(resp_args[:2])
        if resp_args
        else "The respondent defended the lower court order, asserting that all statutory requirements were satisfied."
    )

    # Reasoning & Verdict
    court_thought = (
        " ".join(reasoning[:2])
        if reasoning
        else "The court examined the statutory records and relevant precedents to evaluate the merits."
    )
    final_verdict = (
        " ".join(decision[:2])
        if decision
        else outcome_summary or "The court disposed of the appeal in accordance with its findings."
    )

    return {
        "title": f"{metadata.get('petitioner') or 'Appellant'} v. {metadata.get('respondent') or 'Respondent'}",
        "court": metadata.get("court") or "Supreme / High Court of India",
        "case_no": metadata.get("case_no") or "Appellate Jurisdiction",
        "bench": metadata.get("bench") or "Hon'ble Bench",
        "what_happened": fact_text,
        "core_dispute": issue_text,
        "petitioner_stance": pet_text,
        "respondent_stance": resp_text,
        "court_reasoning": court_thought,
        "final_outcome": final_verdict,
        "statutes": top_statutes[:5]
    }
