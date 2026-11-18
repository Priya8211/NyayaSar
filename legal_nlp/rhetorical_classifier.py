"""
Rhetorical Role Classifier for Indian Court Judgments.
Supports the 6 LegalSeg categories:
1. Facts
2. Issue
3. Arguments of Petitioner
4. Arguments of Respondent
5. Reasoning
6. Decision

Features:
- Seamlessly loads pre-trained scikit-learn models (best_role_classifier.pkl).
- Intelligent Contextual Fallback Engine that combines:
  a) Positional document priors (Facts at start, Decision at end).
  b) Strong Indian legal discourse markers.
  c) Keyword lexical matching.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

# Supported labels
LEGAL_ROLES = [
    "Facts",
    "Issue",
    "Arguments of Petitioner",
    "Arguments of Respondent",
    "Reasoning",
    "Decision"
]

ROLE_COLORS = {
    "Facts": "#1E88E5",                  # Blue
    "Issue": "#00ACC1",                  # Cyan
    "Arguments of Petitioner": "#FB8C00", # Amber
    "Arguments of Respondent": "#F4511E", # Deep Orange
    "Reasoning": "#8E24AA",              # Purple
    "Decision": "#43A047"                # Green
}

ROLE_DESCRIPTIONS = {
    "Facts": "Background events, chronological history, parties involved, and lower court proceedings.",
    "Issue": "The central legal dispute or points of law formulated for determination by the court.",
    "Arguments of Petitioner": "Legal submissions, pleas, and contentions raised by the appellant or petitioner.",
    "Arguments of Respondent": "Submissions, counter-arguments, and defense raised by the respondent or State.",
    "Reasoning": "The court's analytical evaluation of statutory provisions, evidence, and binding precedents.",
    "Decision": "The final judicial verdict, formal operative disposition, and relief granted or denied."
}


class RhetoricalClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.labels = LEGAL_ROLES
        self.is_custom_model = False
        self.model_info = {"model_name": "Contextual Legal Discourse Engine (Hybrid)"}

        # Try to load saved model from disk if available
        candidate_paths = [
            model_path,
            "best_role_classifier.pkl",
            os.path.join(os.path.dirname(__file__), "..", "best_role_classifier.pkl")
        ]
        
        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    import joblib
                    self.model = joblib.load(path)
                    self.is_custom_model = True
                    self.model_info["model_name"] = "Trained Linear SVM (LegalSeg)"
                    print(f"Loaded pre-trained rhetorical classifier from {path}")
                    break
                except Exception as e:
                    print(f"Warning: Could not load model from {path}: {e}")

    def predict_sentences(self, sentences: List[str]) -> pd.DataFrame:
        """
        Predicts rhetorical roles for a list of sentences, returning a DataFrame with:
        Sentence, Predicted Role, Confidence, and Color.
        """
        if not sentences:
            return pd.DataFrame(columns=["Sentence", "Predicted Role", "Confidence", "Color"])

        if self.is_custom_model and self.model is not None:
            try:
                preds = self.model.predict(sentences)
                # Compute confidence / decision scores
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(sentences).max(axis=1)
                elif hasattr(self.model, "decision_function"):
                    margins = self.model.decision_function(sentences)
                    if margins.ndim == 1:
                        probs = 1.0 / (1.0 + np.exp(-margins))
                    else:
                        # Softmax over decision function
                        exp_m = np.exp(margins - margins.max(axis=1, keepdims=True))
                        probs = (exp_m / exp_m.sum(axis=1, keepdims=True)).max(axis=1)
                else:
                    probs = [0.85] * len(sentences)

                df = pd.DataFrame({
                    "Sentence": sentences,
                    "Predicted Role": preds,
                    "Confidence": np.round(probs, 3)
                })
                df["Color"] = df["Predicted Role"].map(ROLE_COLORS).fillna("#757575")
                return df
            except Exception as e:
                print(f"Prediction using loaded model failed ({e}), falling back to Contextual Engine.")

        # High-Precision Contextual & Positional Engine
        results = []
        n_sentences = max(1, len(sentences))

        for idx, sent in enumerate(sentences):
            norm_pos = idx / n_sentences  # 0.0 at beginning, 1.0 at end
            role, conf = self._classify_single_contextual(sent, norm_pos)
            results.append({
                "Sentence": sent,
                "Predicted Role": role,
                "Confidence": conf,
                "Color": ROLE_COLORS.get(role, "#757575")
            })

        return pd.DataFrame(results)

    def _classify_single_contextual(self, sentence: str, norm_pos: float) -> Tuple[str, float]:
        """
        Classifies a single legal sentence using rhetorical discourse cues,
        linguistic markers, and sentence document position priors.
        """
        s = sentence.lower().strip()

        # 1. DECISION MARKERS (Strongest at the end of document)
        if (
            re.search(
                r"\b(?:accordingly|in the result|in view of the above|for the reasons stated|in the light of the above|consequently)\b.*"
                r"\b(?:appeal is allowed|appeal is dismissed|petition is allowed|petition is dismissed|impugned order is set aside|conviction is set aside|appellant is acquitted|stands disposed of|no order as to costs)\b",
                s
            )
            or re.search(r"\b(?:appeal is (?:hereby )?(?:allowed|dismissed)|petition is (?:hereby )?(?:allowed|dismissed))\b", s)
            or re.search(r"\b(?:impugned judgment and order of conviction is set aside|appellant is acquitted of all charges)\b", s)
            or (norm_pos > 0.85 and re.search(r"\b(?:bail is granted|sentenced to|ordered accordingly|costs)\b", s))
        ):
            return "Decision", 0.94

        # 2. ISSUE MARKERS (Formulation of points of determination)
        if (
            re.search(
                r"\b(?:the (?:short |sole |pivotal |core |main )?question (?:that |which )?(?:arises|arising|for consideration)|"
                r"the issue for our determination|point for determination|whether the high court|whether the prosecution|"
                r"whether in the facts and circumstances|the following question of law arises)\b",
                s
            )
            or (s.startswith("whether ") and "?" in s)
        ):
            return "Issue", 0.92

        # 3. ARGUMENTS OF PETITIONER
        if re.search(
            r"\b(?:learned counsel (?:appearing )?for the (?:appellant|petitioner)|"
            r"mr\..*learned senior counsel for the appellant|"
            r"submitted by the learned counsel for the petitioner|"
            r"contended on behalf of the appellant|"
            r"the appellant submits that|petitioner argued that|the appellant contended that|"
            r"learned counsel for the convict)\b",
            s
        ):
            return "Arguments of Petitioner", 0.91

        # 4. ARGUMENTS OF RESPONDENT
        if re.search(
            r"\b(?:learned counsel (?:appearing )?for the (?:respondent|state|prosecution)|"
            r"learned additional solicitor general|learned standing counsel for the state|"
            r"on the other hand.*submitted|learned counsel for the respondents contended|"
            r"the state contended that|the respondent submits that|opposed the prayer for bail)\b",
            s
        ):
            return "Arguments of Respondent", 0.91

        # 5. REASONING MARKERS (Judicial analysis, precedent application, statutory interpretation)
        if (
            re.search(
                r"\b(?:in our (?:considered )?(?:view|opinion)|we (?:find|hold|are of the view|have no hesitation)|"
                r"it is well settled (?:that|law)|it is well-established|as held by this court|"
                r"a perusal of section|on a reading of article|applying the ratio of|"
                r"we are unable to accept|the principle of|we are in agreement with|"
                r"reliance was placed on|the high court failed to appreciate|"
                r"there is no evidence to show|circumstantial evidence establishes)\b",
                s
            )
        ):
            return "Reasoning", 0.88

        # 6. FACTS MARKERS (Events, lower court history, FIR, witnesses)
        if (
            re.search(
                r"\b(?:the appellant was convicted|the prosecution case is that|according to the prosecution|"
                r"an fir was registered|the deceased was|the incident occurred on|the complainant filed|"
                r"charges were framed|the trial court convicted|pw-?\d+|exhibit p-?\d+|post-mortem report|"
                r"the present appeal arises out of|the facts of the case|the appellant filed an application)\b",
                s
            )
        ):
            return "Facts", 0.89

        # POSITIONAL BACKOFF HEURISTIC
        if norm_pos < 0.25:
            return "Facts", 0.72
        elif 0.25 <= norm_pos < 0.50:
            if "argue" in s or "submit" in s or "plead" in s:
                return "Arguments of Petitioner", 0.68
            return "Facts", 0.65
        elif 0.50 <= norm_pos < 0.85:
            return "Reasoning", 0.75
        else:
            if "allow" in s or "dismiss" in s or "set aside" in s or "directed" in s or "result" in s:
                return "Decision", 0.82
            return "Reasoning", 0.70
