"""
Streamlit Application: Indian Court Judgment Analyzer & Simplifier.
An advanced NLP system to simplify, segment, summarize, and demystify Indian judicial decisions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure local package import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from legal_nlp.preprocessor import (
    legal_sent_tokenize,
    extract_document_header,
    extract_text_from_upload,
    clean_legal_text
)
from legal_nlp.rhetorical_classifier import (
    RhetoricalClassifier,
    ROLE_COLORS,
    ROLE_DESCRIPTIONS,
    LEGAL_ROLES
)
from legal_nlp.summarizer import LegalSummarizer
from legal_nlp.plain_english import (
    detect_and_explain_jargon,
    lookup_statutory_sections,
    calculate_readability,
    generate_layman_brief
)
from legal_nlp.legal_ner import (
    extract_all_entities,
    determine_bench_type,
    extract_verdict_outcome
)
from legal_nlp.sample_judgments import SAMPLE_JUDGMENTS


# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="NyayaSar | Indian Court Judgment Simplifier",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sleek CSS
st.markdown("""
<style>
    /* Global Font & Header */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    
    /* Card Containers */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .verdict-banner-green {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 1px solid #6ee7b7;
        color: #065f46;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .verdict-banner-red {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 1px solid #fca5a5;
        color: #991b1b;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .verdict-banner-blue {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1px solid #93c5fd;
        color: #1e40af;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    /* Role Badges */
    .role-badge {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin-right: 6px;
    }
    .sentence-box {
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 5px solid;
        background: #fdfdfd;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        line-height: 1.6;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. INITIALIZE CACHED NLP ENGINES
# ==============================================================================

@st.cache_resource
def get_classifier():
    return RhetoricalClassifier()

@st.cache_resource
def get_summarizer():
    return LegalSummarizer()

classifier = get_classifier()
summarizer = get_summarizer()


# ==============================================================================
# 3. SIDEBAR CONTROLS & INPUT HANDLING
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scales.png", width=64)
    st.title("NyayaSar (न्यायसार)")
    st.caption("AI-Powered Indian Court Judgment Simplifier & Analyzer")
    st.divider()

    st.subheader("1. Document Input")
    input_source = st.radio(
        "Choose Judgment Source:",
        ["📂 Pre-Loaded Landmark Cases", "📤 Upload File (PDF/TXT)", "✍️ Paste Judgment Text"],
        index=0
    )

    judgment_text = ""
    case_name_label = ""

    if input_source == "📂 Pre-Loaded Landmark Cases":
        selected_case = st.selectbox(
            "Select Landmark Judgment:",
            list(SAMPLE_JUDGMENTS.keys()),
            index=0
        )
        judgment_text = SAMPLE_JUDGMENTS[selected_case]
        case_name_label = selected_case

    elif input_source == "📤 Upload File (PDF/TXT)":
        uploaded_file = st.file_uploader(
            "Upload Indian Court Judgment",
            type=["txt", "pdf"],
            help="Upload supreme court or high court judgment in text or PDF format."
        )
        if uploaded_file is not None:
            judgment_text = extract_text_from_upload(uploaded_file)
            case_name_label = uploaded_file.name

    else:
        pasted = st.text_area(
            "Paste Judgment Text here:",
            height=250,
            placeholder="IN THE SUPREME COURT OF INDIA..."
        )
        judgment_text = pasted
        case_name_label = "Custom Pasted Judgment"

    st.divider()

    # Summarization Settings
    st.subheader("2. Summarization Controls")
    summary_len = st.slider(
        "Summary Sentences:",
        min_value=3,
        max_value=12,
        value=5,
        step=1,
        help="Target number of representative sentences for the extractive legal summary."
    )
    use_role_boost = st.checkbox(
        "Boost Decision & Reasoning Roles",
        value=True,
        help="Gives higher weight to judicial reasoning and holding while down-weighting procedural boilerplate."
    )

    st.divider()

    # Model status
    st.subheader("3. Model Status")
    st.success(f"Active Engine: {classifier.model_info['model_name']}")
    st.caption("Dataset: LegalSeg (Indian Rhetorical Roles)")
    st.caption("6 Roles: Facts, Issue, Arguments (Petitioner/Respondent), Reasoning, Decision.")


# ==============================================================================
# 4. MAIN PAGE DISPLAY & PIPELINE EXECUTION
# ==============================================================================

st.markdown('<div class="main-title">⚖️ Indian Court Judgment Simplifier & Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Transforming dense judicial legalese into plain English through '
    'rhetorical segmentation, legal entity extraction, and role-weighted summarization.</div>',
    unsafe_allow_html=True
)

if not judgment_text or len(judgment_text.strip()) < 50:
    st.info("👈 Please select a pre-loaded landmark case, upload a PDF/TXT file, or paste text in the sidebar to begin analysis.")
    st.stop()

# Clean & Tokenize
with st.spinner("Analyzing legal judgment, classifying rhetorical roles, and extracting legal knowledge..."):
    clean_text = clean_legal_text(judgment_text)
    sentences = legal_sent_tokenize(clean_text)

    if not sentences:
        st.error("Could not extract coherent sentences from the provided text. Please check the document format.")
        st.stop()

    # Execute NLP Pipeline
    metadata = extract_document_header(clean_text)
    df_results = classifier.predict_sentences(sentences)
    entities = extract_all_entities(clean_text)
    outcome = extract_verdict_outcome(clean_text)
    bench_info = determine_bench_type(entities.get("Judges", []))
    statutes_lookup = lookup_statutory_sections(clean_text)
    jargon_detected = detect_and_explain_jargon(clean_text)
    readability_orig = calculate_readability(clean_text)

    # Summarization
    extractive_summary = summarizer.summarize(
        sentences,
        roles=df_results["Predicted Role"].tolist(),
        num_sentences=summary_len,
        use_role_weights=use_role_boost
    )
    role_summaries = summarizer.summarize_by_role(df_results, sentences_per_role=2)

    # Group sentences by role for layman brief
    role_grouped = {}
    for r in LEGAL_ROLES:
        role_grouped[r] = df_results[df_results["Predicted Role"] == r]["Sentence"].tolist()

    layman_brief = generate_layman_brief(
        metadata=metadata,
        role_sentences=role_grouped,
        outcome_summary=outcome["summary"],
        top_statutes=statutes_lookup
    )

    # Readability of the simplified brief
    brief_text = f"{layman_brief['what_happened']} {layman_brief['core_dispute']} {layman_brief['court_reasoning']} {layman_brief['final_outcome']}"
    readability_brief = calculate_readability(brief_text)


# ==============================================================================
# 5. DASHBOARD TABS
# ==============================================================================

tab_brief, tab_reader, tab_jargon, tab_analytics, tab_summary, tab_export = st.tabs([
    "📋 Case at a Glance (Plain English)",
    "📖 Interactive Document Reader",
    "💡 Plain-English Jargon Buster",
    "📊 NLP & Readability Analytics",
    "📑 Extractive Summaries",
    "📥 Export & Brief Generator"
])


# ------------------------------------------------------------------------------
# TAB 1: CASE AT A GLANCE (EXECUTIVE PLAIN-ENGLISH BRIEF)
# ------------------------------------------------------------------------------
with tab_brief:
    # 1. Verdict Banner
    banner_cls = (
        "verdict-banner-green" if outcome["badge_color"] == "green"
        else "verdict-banner-red" if outcome["badge_color"] == "red"
        else "verdict-banner-blue"
    )
    st.markdown(f"""
    <div class="{banner_cls}">
        <h3 style="margin: 0; font-weight: 700;">⚖️ Final Outcome: {outcome['status']}</h3>
        <p style="margin: 6px 0 0 0; font-size: 1.05rem;">{outcome['summary']}</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Case Metadata KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Court", metadata.get("court") or (entities.get("Courts", ["Court"])[0] if entities.get("Courts") else "Supreme Court of India"))
    with col2:
        st.metric("Bench Strength", bench_info["type"], f"{bench_info['count']} Judge(s)")
    with col3:
        st.metric("Total Sentences", len(sentences), f"{len(clean_text.split())} words")
    with col4:
        st.metric("Rhetorical Roles", f"{df_results['Predicted Role'].nunique()} / 6 Detected")

    st.write("")

    # 3. Readability Reduction Impact
    st.subheader("📉 Simplification & Readability Reduction")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#dc2626;">Original Judgment Complexity</h4>
            <p style="font-size:1.8rem; font-weight:700; margin:5px 0;">Grade {readability_orig['flesch_kincaid_grade']}</p>
            <span style="color:#64748b; font-size:0.85rem;">{readability_orig['level_label']}</span><br>
            <small>Avg sentence length: <b>{readability_orig['avg_sentence_length']} words</b></small>
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#16a34a;">Simplified Brief Readability</h4>
            <p style="font-size:1.8rem; font-weight:700; margin:5px 0;">Grade {readability_brief['flesch_kincaid_grade']}</p>
            <span style="color:#64748b; font-size:0.85rem;">{readability_brief['level_label']}</span><br>
            <small>Avg sentence length: <b>{readability_brief['avg_sentence_length']} words</b></small>
        </div>
        """, unsafe_allow_html=True)

    with col_r3:
        grade_drop = max(0.0, round(readability_orig['flesch_kincaid_grade'] - readability_brief['flesch_kincaid_grade'], 1))
        ease_gain = max(0.0, round(readability_brief['flesch_reading_ease'] - readability_orig['flesch_reading_ease'], 1))
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#2563eb;">Accessibility Gain</h4>
            <p style="font-size:1.8rem; font-weight:700; margin:5px 0;">+{ease_gain} pts</p>
            <span style="color:#64748b; font-size:0.85rem;">Flesch Reading Ease Improvement</span><br>
            <small>Complexity reduced by: <b>{grade_drop} grade levels</b></small>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    # 4. Structured Plain-English 5-Point Brief
    st.subheader("📝 5-Point Plain English Case Brief")
    
    with st.container():
        st.markdown("#### 1. 📌 What Happened? (The Facts in Simple Terms)")
        st.info(layman_brief["what_happened"])

        st.markdown("#### 2. ⚖️ What Was the Core Legal Dispute? (The Issue)")
        st.warning(layman_brief["core_dispute"])

        col_arg1, col_arg2 = st.columns(2)
        with col_arg1:
            st.markdown("#### 3. 🗣️ What Did the Petitioner Argue?")
            st.markdown(f"<div style='background:#fffbeb; padding:12px; border-radius:8px; border-left:4px solid #f59e0b;'>{layman_brief['petitioner_stance']}</div>", unsafe_allow_html=True)

        with col_arg2:
            st.markdown("#### 4. 🛡️ What Did the Respondent Argue?")
            st.markdown(f"<div style='background:#fff7ed; padding:12px; border-radius:8px; border-left:4px solid #ea580c;'>{layman_brief['respondent_stance']}</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown("#### 5. 🧠 Why Did the Court Decide This Way? (Judicial Reasoning & Ratio)")
        st.markdown(f"<div style='background:#faf5ff; padding:14px; border-radius:8px; border-left:4px solid #9333ea;'>{layman_brief['court_reasoning']}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# TAB 2: INTERACTIVE DOCUMENT READER & COLOR-CODED SEGMENTATION
# ------------------------------------------------------------------------------
with tab_reader:
    st.subheader("📖 Rhetorical Role Segmented Document")
    st.write("Each sentence has been categorized into one of the 6 standard rhetorical roles. Toggle or filter to inspect specific legal elements.")

    # Role Legend with counts
    legend_cols = st.columns(6)
    role_counts = df_results["Predicted Role"].value_counts().to_dict()

    for idx, (role, color) in enumerate(ROLE_COLORS.items()):
        with legend_cols[idx]:
            count = role_counts.get(role, 0)
            st.markdown(
                f"<div style='background:{color}; color:white; padding:6px 10px; border-radius:6px; text-align:center; font-size:0.82rem; font-weight:600;'>"
                f"{role}<br><span style='font-size:1.1rem;'>{count}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.write("")

    # Filter selector
    selected_filter = st.multiselect(
        "Filter visible roles in document:",
        options=LEGAL_ROLES,
        default=LEGAL_ROLES
    )

    filtered_sentences = df_results[df_results["Predicted Role"].isin(selected_filter)]

    # View Mode Toggle
    view_mode = st.radio("Display Mode:", ["Visual Annotated Reader", "Classification Data Table"], horizontal=True)

    if view_mode == "Visual Annotated Reader":
        st.markdown("<div style='max-height: 600px; overflow-y: auto; padding: 10px; border: 1px solid #e2e8f0; border-radius: 8px;'>", unsafe_allow_html=True)
        for _, row in filtered_sentences.iterrows():
            role = row["Predicted Role"]
            color = row["Color"]
            conf = row["Confidence"]
            sent = row["Sentence"]

            st.markdown(
                f"<div class='sentence-box' style='border-left-color: {color};'>"
                f"<span class='role-badge' style='background: {color};'>{role}</span>"
                f"<span style='color: #64748b; font-size: 0.78rem; font-weight: 500;'>Conf: {int(conf*100)}%</span><br>"
                f"<span style='font-size: 0.95rem; color: #1e293b;'>{sent}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.dataframe(
            filtered_sentences[["Sentence", "Predicted Role", "Confidence"]],
            use_container_width=True,
            height=500
        )


# ------------------------------------------------------------------------------
# TAB 3: PLAIN-ENGLISH JARGON BUSTER & STATUTE EXPLAINER
# ------------------------------------------------------------------------------
with tab_jargon:
    st.subheader("💡 Demystifying Legal Terminology & Statutory Citations")
    st.write("Court judgments are heavily saturated with Latin maxims and technical provisions. NyayaSar automatically translates them into plain English.")

    col_j1, col_j2 = st.columns(2)

    with col_j1:
        st.markdown("### 🏛️ Latin Maxims & Legal Jargon Detected")
        if jargon_detected:
            st.caption(f"Found {len(jargon_detected)} unique technical legal terms:")
            for item in jargon_detected:
                with st.expander(f"📌 {item['term']} ({item['occurrences']}x found)"):
                    st.markdown(f"**Category:** `{item['category']}`")
                    st.markdown(f"**Plain-English Meaning:** {item['plain_meaning']}")
        else:
            st.info("No common Latin maxims detected in this excerpt.")

    with col_j2:
        st.markdown("### 📜 Statutory Sections & Constitutional Articles")
        if statutes_lookup:
            st.caption(f"Identified {len(statutes_lookup)} substantive provisions:")
            for item in statutes_lookup:
                with st.expander(f"⚖️ {item['citation']}"):
                    st.markdown(f"**What it means:** {item['meaning']}")
        else:
            st.info("No specific statutory sections matched our Indian legal knowledge base.")


# ------------------------------------------------------------------------------
# TAB 4: NLP & READABILITY ANALYTICS
# ------------------------------------------------------------------------------
with tab_analytics:
    st.subheader("📊 Document Structure & Rhetorical Analytics")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # Donut Chart of Rhetorical Roles
        role_df = df_results["Predicted Role"].value_counts().reset_index()
        role_df.columns = ["Role", "Count"]

        fig_donut = px.pie(
            role_df,
            names="Role",
            values="Count",
            color="Role",
            color_discrete_map=ROLE_COLORS,
            hole=0.45,
            title="Rhetorical Role Composition"
        )
        fig_donut.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=350)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_g2:
        # Rhetorical Flow Timeline across Document
        df_flow = df_results.copy()
        df_flow["Position (%)"] = np.linspace(0, 100, len(df_flow))

        fig_flow = px.scatter(
            df_flow,
            x="Position (%)",
            y="Predicted Role",
            color="Predicted Role",
            color_discrete_map=ROLE_COLORS,
            title="Rhetorical Flow: Sentence Position vs Role",
            labels={"Predicted Role": "Rhetorical Role"}
        )
        fig_flow.update_layout(
            margin=dict(t=40, b=0, l=0, r=0),
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_flow, use_container_width=True)

    st.divider()

    # Named Entities Breakdown
    st.subheader("🏷️ Legal Named Entities Extracted")
    ent_cols = st.columns(3)

    with ent_cols[0]:
        st.markdown("#### 📜 Central Statutes & Acts")
        statutes = entities.get("Statutes & Acts", [])
        if statutes:
            for s in statutes:
                st.write(f"• **{s}**")
        else:
            st.caption("None detected")

    with ent_cols[1]:
        st.markdown("#### ⚖️ Courts & Authorities")
        courts = entities.get("Courts", [])
        if courts:
            for c in courts:
                st.write(f"• {c}")
        else:
            st.caption("None detected")

    with ent_cols[2]:
        st.markdown("#### 👨‍⚖️ Judges & Case Citations")
        judges = entities.get("Judges", [])
        citations = entities.get("Case Citations", [])
        if judges:
            st.write(f"**Judges:** {', '.join(judges)}")
        if citations:
            st.write(f"**Citations:** {', '.join(citations)}")
        if not judges and not citations:
            st.caption("None detected")


# ------------------------------------------------------------------------------
# TAB 5: EXTRACTIVE & ROLE-BASED SUMMARIES
# ------------------------------------------------------------------------------
with tab_summary:
    st.subheader(f"📑 Extractive Summary ({len(extractive_summary)} Key Sentences)")
    st.caption("Generated using Legal-Weighted TextRank with Maximal Marginal Relevance (MMR) for diversity.")

    for item in extractive_summary:
        role = item["role"]
        color = ROLE_COLORS.get(role, "#64748b")
        st.markdown(
            f"<div class='sentence-box' style='border-left-color: {color};'>"
            f"<span style='font-weight: 700; color: #64748b; margin-right: 8px;'>#{item['rank']}</span>"
            f"<span class='role-badge' style='background: {color};'>{role}</span>"
            f"<span style='color: #64748b; font-size: 0.8rem;'>Score: {item['score']}</span><br>"
            f"<span style='font-size: 0.98rem; color: #1e293b;'>{item['sentence']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.write("")
    st.divider()

    st.subheader("📂 Role-Based Summaries")
    st.write("Inspect the top representative sentences partitioned by each rhetorical section:")

    for role in LEGAL_ROLES:
        role_sents = role_summaries.get(role, [])
        color = ROLE_COLORS.get(role, "#475569")
        if role_sents:
            with st.expander(f"{role} ({len(role_sents)} key points)"):
                for s in role_sents:
                    st.markdown(f"• {s['sentence']}")


# ------------------------------------------------------------------------------
# TAB 6: EXPORT & LEGAL REPORT GENERATOR
# ------------------------------------------------------------------------------
with tab_export:
    st.subheader("📥 Export Legal Brief & Analysis Data")
    st.write("Download formatted case summaries or sentence classification datasets for academic reporting or client briefing.")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        # Markdown Brief Generator
        markdown_report = f"""# EXECUTIVE LEGAL BRIEF: {layman_brief['title']}
**Court:** {layman_brief['court']}  
**Bench:** {layman_brief['bench']}  
**Final Outcome:** {outcome['status']}  

---

## 1. Executive Summary
{outcome['summary']}

## 2. Background & Facts
{layman_brief['what_happened']}

## 3. Core Legal Issue
{layman_brief['core_dispute']}

## 4. Submissions
- **Petitioner / Appellant:** {layman_brief['petitioner_stance']}
- **Respondent / State:** {layman_brief['respondent_stance']}

## 5. Judicial Reasoning & Holding
{layman_brief['court_reasoning']}

## 6. Final Verdict & Order
{layman_brief['final_outcome']}

---

### Key Statutory Provisions Cited:
"""
        for item in statutes_lookup:
            markdown_report += f"- **{item['citation']}:** {item['meaning']}\n"

        markdown_report += f"""
---
### Readability & Accessibility Metrics:
- **Original Judgment:** Grade {readability_orig['flesch_kincaid_grade']} (Flesch Ease: {readability_orig['flesch_reading_ease']})
- **Simplified Brief:** Grade {readability_brief['flesch_kincaid_grade']} (Flesch Ease: {readability_brief['flesch_reading_ease']})
- **Complexity Reduction:** {grade_drop} school grade levels

*Generated via NyayaSar NLP Judgment Analyzer.*
"""
        st.download_button(
            label="📄 Download Executive Brief (Markdown)",
            data=markdown_report,
            file_name="executive_legal_brief.md",
            mime="text/markdown"
        )

    with col_e2:
        # CSV Export
        csv_data = df_results[["Sentence", "Predicted Role", "Confidence"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📊 Download Sentence Classification (CSV)",
            data=csv_data,
            file_name="judgment_rhetorical_analysis.csv",
            mime="text/csv"
        )
