# NyayaSar (न्यायसार) ⚖️
### AI-Powered Indian Court Judgment Analyzer & Simplifier

> **Problem Statement:** *"To simplify Indian court judgments using NLP-based analysis and summarization."*

---

## 📌 Project Overview
Indian court judgments are notoriously lengthy, written in archaic colonial legalese, and filled with dense statutory citations and Latin maxims (*res judicata*, *mens rea*, *inter alia*, *ratio decidendi*). For common citizens, law students, and litigants, understanding what actually happened, what the legal issues were, and why the court arrived at its verdict is nearly impossible without costly legal assistance.

**NyayaSar** bridges this gap using an advanced Natural Language Processing (NLP) pipeline that transforms complex judicial texts into clear, actionable, plain-English executive summaries and interactive visual intelligence.

---

## 🚀 Key Enhancements & Features

### 1. 📝 Plain-English Simplification & Layman Brief
- **Latin Maxim & Legal Jargon Buster**: Built-in glossary that detects and explains 70+ Latin legal maxims and technical Indian legal terms (*prima facie*, *suo motu*, *habeas corpus*, *certiorari*, *mandamus*, *in limine*, *quash*, *cognizable*).
- **Statutory Section Explainer**: Automatically looks up cited sections of the **Indian Penal Code (IPC)**, **Code of Criminal Procedure (CrPC)**, and **Constitution of India** (e.g., Section 302 IPC, Section 420 IPC, Article 21) and translates them into plain-language meanings.
- **5-Point Layman Brief**: Automatically synthesizes answers to the 5 core citizen questions:
  1. *What happened?* (Facts in simple terms)
  2. *What was the core dispute?* (Issue framed as a straightforward question)
  3. *What did the Petitioner argue?*
  4. *What did the Respondent argue?*
  5. *Why did the Court rule this way?* (Judicial reasoning & final outcome)

### 2. 📉 Quantifiable Readability Scoring
- Measures reading complexity **before and after simplification** using:
  - **Flesch-Kincaid Grade Level**
  - **Flesch Reading Ease Score**
  - **Gunning Fog Index**
  - **Average Sentence Length** & **Percentage of Complex Words**
- Demonstrates complexity reduction (e.g., from Grade 18+ Doctorate/Senior Judicial down to Grade 9–10 High School level).

### 3. ⚖️ Rhetorical Role Classification (LegalSeg)
- Segments judgment sentences into the 6 standard Indian legal rhetorical roles:
  1. **Facts**: Chronological background, events, trial history.
  2. **Issue**: Core questions of law formulated for determination.
  3. **Arguments of Petitioner**: Pleas and submissions by the appellant.
  4. **Arguments of Respondent**: Contentions and defenses by the State/respondent.
  5. **Reasoning**: Judicial evaluation of evidence, doctrine, and precedents.
  6. **Decision**: Operative final verdict, orders, and relief granted/denied.
- Supports trained scikit-learn models (`best_role_classifier.pkl`) with a high-precision **Contextual Legal Discourse Engine** combining positional document priors and linguistic discourse markers.

### 4. 🧠 Legal-Weighted TextRank Summarization
- Extractive summarization using cosine similarity on TF-IDF sentence representations with **PageRank centrality**.
- **Role Weighting**: Dynamically boosts holding/decision ($2.0\times$) and reasoning ($1.6\times$) sentences over procedural boilerplate.
- **Maximal Marginal Relevance (MMR)**: Eliminates repetitive or redundant sentences.
- **Role-Based Summaries**: Extracts top representative sentences grouped by rhetorical category.

### 5. 🏷️ Indian Legal Named Entity Recognition (NER)
- Central Acts & Statutes (IPC, CrPC, CPC, Evidence Act, NDPS, POCSO, Constitution of India).
- Judicial Authorities (Supreme Court, High Courts, District Courts, Tribunals).
- Judges & Bench Strength Detection (Single, Division, Full, Constitution Bench).
- Landmark Precedent Citations (SCC, AIR, SCR, MANU).
- Judicial Outcome / Disposition Classifier (Allowed, Dismissed, Remanded, Disposed).

### 6. 💻 Modern Interactive Streamlit Dashboard
- **1-Click Pre-loaded Landmark Indian Judgments**: Instant testing with authentic Criminal, Constitutional, and Commercial law cases.
- **Visual Annotated Document Reader**: Color-coded sentence reader with real-time confidence scores and role toggles.
- **Interactive Visualizations (Plotly)**: Donut charts of role composition and document rhetorical flow timelines.
- **One-Click Export**: Download formatted Executive Legal Brief (Markdown) and classification results (CSV).

---

## 🛠️ Project Structure
```
NLP Project/
├── app.py                     # Modern Streamlit Dashboard
├── requirements.txt           # Project Dependencies
├── README.md                  # Documentation
├── NLP_Project2 (2).ipynb     # Model Training & Experimentation Notebook
└── legal_nlp/                 # Modular Python NLP Package
    ├── __init__.py
    ├── preprocessor.py        # Legal tokenization & text cleaning
    ├── plain_english.py       # Jargon buster, readability & brief generator
    ├── rhetorical_classifier.py# 6-class rhetorical role classifier
    ├── summarizer.py          # Legal-Weighted TextRank & MMR summarizer
    ├── legal_ner.py           # Indian Legal NER & outcome classifier
    └── sample_judgments.py    # Pre-loaded landmark Indian judgments
```

---

## 🚀 How to Run the Dashboard Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Streamlit:**
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser:**
   Navigate to `http://localhost:8501`.

---

## 📚 Dataset Reference
- Trained on **LegalSeg** (`L-NLProc/LegalSeg_CSV`): Rhetorical role labeling dataset for Indian court judgments.

# NyayaSar-AI
An AI-powered Indian Court Judgment Simplifier and Analyzer that transforms complex legal judgments into easy-to-understand language using rhetorical segmentation, legal entity extraction, readability analysis, role-weighted summarization, and automated brief generation.
