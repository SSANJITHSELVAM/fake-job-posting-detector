"""
inference.py
------------
Steps 5 & 6 of the pipeline: Explainability + Prediction + Risk Signals.

Features:
  1. High-accuracy prediction with calibrated probabilities & risk tiers
  2. Rule-based Red Flag & Legitimacy Signal Auditor
  3. Ultra-fast Token Saliency attribution (<200ms batched inference)
  4. Deep SHAP Text Explainer (Partition algorithm) for rigorous attribution
"""

import html
import logging
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import shap
from xgboost import XGBClassifier

from . import config
from .data_preprocessing import clean_text
from .feature_extraction import RobertaFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scam Pattern & Red Flag Definitions
# ---------------------------------------------------------------------------
RED_FLAG_PATTERNS = [
    {
        "category": "Upfront Payment / Financial Asks",
        "regex": r"\b(wire transfer|western union|moneygram|processing fee|registration fee|starter kit|security deposit|buy equipment|personal bank account|send money|cashapp|venmo|zelle|crypto|bitcoin|gift card)\b",
        "severity": "CRITICAL",
        "description": "Legitimate employers never ask candidates to pay fees, purchase gift cards, or wire money upfront.",
    },
    {
        "category": "Sensitive Personal Data Request",
        "regex": r"\b(social security number|ssn|credit card|debit card|bank routing|driver\'?s license copy|passport copy|bank details)\b",
        "severity": "HIGH",
        "description": "Asking for confidential financial or identity info prior to a formal interview or background check is a major indicator of fraud.",
    },
    {
        "category": "Unofficial Communication Channel",
        "regex": r"\b(whatsapp|telegram|signal|google hangout|skype only|@gmail\.com|@yahoo\.com|@hotmail\.com|@outlook\.com)\b",
        "severity": "HIGH",
        "description": "Recruiters conducting interviews exclusively via WhatsApp/Telegram or using free email domains rather than corporate domains.",
    },
    {
        "category": "Unrealistic Compensation / Easy Money",
        "regex": r"\b(\$\d{4,}\s*(per|\/)\s*week|earn\s*\$\d{3,}\s*(per|\/)\s*day|no experience needed|no qualifications|work 1 hour|guaranteed income|100\%\s*guaranteed|get rich|easy data entry)\b",
        "severity": "MEDIUM",
        "description": "Extravagant compensation promises paired with zero requirement or minimal work hours.",
    },
    {
        "category": "Urgency / Pressure Tactics",
        "regex": r"\b(urgent|hurry|act now|apply immediately|limited spots|today only|instant hire|no interview required|guaranteed hire)\b",
        "severity": "MEDIUM",
        "description": "Creating artificial urgency or bypassing standard interview stages to induce hasty applicant decisions.",
    },
    {
        "category": "Check Cashing / Reshipment Scheme",
        "regex": r"\b(mystery shopper|secret shopper|package inspector|package forwarding|reshipment|deposit check|cashier\'?s check|excess funds)\b",
        "severity": "CRITICAL",
        "description": "Classic mule or counterfeit check scams where victims deposit fraudulent checks and wire back 'excess' funds.",
    },
]

LEGITIMACY_PATTERNS = [
    {
        "category": "Standard Benefits Package",
        "regex": r"\b(401\(?k\)?(\s*matching)?|health insurance|dental and vision|paid time off|pto|maternity leave|parental leave|life insurance|disability insurance)\b",
        "description": "Comprehensive institutional employee benefits package.",
    },
    {
        "category": "Structured Technical / Domain Requirements",
        "regex": r"\b(bachelor\'?s degree|master\'?s|ph\.?d|years of experience|proficiency in|familiarity with|rest api|sql|python|java|javascript|react|aws|gcp|docker|kubernetes|clinical experience|rn license|cpa)\b",
        "description": "Concrete, verifiable qualification requirements.",
    },
    {
        "category": "Formal Interview / Hiring Process",
        "regex": r"\b(equal opportunity employer|eoe|background check|drug screen|technical interview|panel interview|portfolio review|affirmative action)\b",
        "description": "Standard corporate hiring and compliance practices.",
    },
]


class FakeJobDetector:
    """Production-grade detector serving predictions, risk signals, and explanations."""

    def __init__(self):
        logger.info("Initializing FakeJobDetector with RoBERTa and XGBoost...")
        self.extractor = RobertaFeatureExtractor()

        self.model = XGBClassifier()
        if os.path.exists(config.XGB_MODEL_PATH):
            logger.info("Loading XGBoost model from %s", config.XGB_MODEL_PATH)
            self.model.load_model(config.XGB_MODEL_PATH)
        else:
            logger.warning("XGBoost model file not found at %s. Prediction will error until trained.", config.XGB_MODEL_PATH)

        self._explainer = None

    # ------------------------------------------------------------------
    # Model Forward Pass
    # ------------------------------------------------------------------
    def _predict_proba_from_raw_texts(self, texts: List[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        features = self.extractor.extract_features(cleaned, verbose=False)
        return self.model.predict_proba(features)

    # ------------------------------------------------------------------
    # Prediction & Risk Tiering
    # ------------------------------------------------------------------
    def predict(self, text: str) -> Dict:
        """Returns the hybrid calibrated label, confidence, risk tier, and probabilities."""
        cleaned = clean_text(text)
        feat = self.extractor.extract_single(cleaned).reshape(1, -1)
        proba = self.model.predict_proba(feat)[0]

        raw_prob_real = float(proba[0])
        raw_prob_fake = float(proba[1])

        # Audit deterministic red flags & legitimacy signals
        audit = self.audit_risk_signals(text)
        
        # Calculate rule-based risk boost
        rule_risk = 0.0
        for rf in audit["red_flags"]:
            if rf["severity"] == "CRITICAL":
                rule_risk += 0.35
            elif rf["severity"] == "HIGH":
                rule_risk += 0.20
            elif rf["severity"] == "MEDIUM":
                rule_risk += 0.10

        # Legitimacy discount
        legit_discount = min(audit["legitimacy_count"] * 0.08, 0.25)
        rule_risk = max(0.0, rule_risk - legit_discount)

        # Blended risk probability
        if audit["red_flag_count"] > 0:
            # If explicit fraud markers are present, calibrate risk upward
            blended_fake = 0.55 * raw_prob_fake + 0.45 * min(1.0, rule_risk + 0.20)
            # If critical red flag (e.g. wire money, crypto, starter kit fee), set minimum risk floor
            has_critical = any(rf["severity"] == "CRITICAL" for rf in audit["red_flags"])
            if has_critical:
                blended_fake = max(blended_fake, 0.75)
        else:
            # When no red flags exist and legitimacy signals are present
            blended_fake = max(0.0, raw_prob_fake - legit_discount)

        blended_fake = float(np.clip(blended_fake, 0.01, 0.99))
        blended_real = float(1.0 - blended_fake)

        label = "Fake" if blended_fake >= 0.50 else "Real"
        confidence = blended_fake if label == "Fake" else blended_real

        # Risk Tier categorization
        if blended_fake >= 0.75:
            risk_tier = "Critical / Severe Risk"
            risk_color = "#ef4444"  # bright crimson
            risk_badge = "🔴 CRITICAL FRAUD RISK"
        elif blended_fake >= 0.45:
            risk_tier = "High Risk (Likely Fake)"
            risk_color = "#f97316"  # orange
            risk_badge = "🟠 SUSPICIOUS / FRAUD RISK"
        elif blended_fake >= 0.20:
            risk_tier = "Moderate Risk (Needs Caution)"
            risk_color = "#eab308"  # amber/yellow
            risk_badge = "🟡 CAUTION ADVISED"
        else:
            risk_tier = "Low Risk (Likely Legitimate)"
            risk_color = "#10b981"  # emerald green
            risk_badge = "🟢 VERIFIED LEGITIMATE"

        return {
            "label": label,
            "confidence": confidence,
            "probability_fake": blended_fake,
            "probability_real": blended_real,
            "raw_model_prob_fake": raw_prob_fake,
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "risk_badge": risk_badge,
            "red_flags": audit["red_flags"],
            "legitimacy_signals": audit["legitimacy_signals"],
            "red_flag_count": audit["red_flag_count"],
            "legitimacy_count": audit["legitimacy_count"],
        }

    # ------------------------------------------------------------------
    # Rule-based Signal & Red Flag Auditor
    # ------------------------------------------------------------------
    def audit_risk_signals(self, text: str) -> Dict:
        """Analyzes text for specific known scam patterns and legitimate corporate signals."""
        lower_text = text.lower()

        detected_red_flags = []
        for pattern in RED_FLAG_PATTERNS:
            matches = list(set(re.findall(pattern["regex"], lower_text, flags=re.IGNORECASE)))
            if matches:
                # Handle regex tuples if groups were returned
                flat_matches = []
                for m in matches:
                    if isinstance(m, tuple):
                        flat_matches.extend([item for item in m if item])
                    else:
                        flat_matches.append(m)
                detected_red_flags.append({
                    "category": pattern["category"],
                    "severity": pattern["severity"],
                    "matches": list(set(flat_matches)),
                    "description": pattern["description"],
                })

        detected_legitimacy = []
        for pattern in LEGITIMACY_PATTERNS:
            matches = list(set(re.findall(pattern["regex"], lower_text, flags=re.IGNORECASE)))
            if matches:
                flat_matches = []
                for m in matches:
                    if isinstance(m, tuple):
                        flat_matches.extend([item for item in m if item])
                    else:
                        flat_matches.append(m)
                detected_legitimacy.append({
                    "category": pattern["category"],
                    "matches": list(set(flat_matches)),
                    "description": pattern["description"],
                })

        return {
            "red_flags": detected_red_flags,
            "legitimacy_signals": detected_legitimacy,
            "red_flag_count": len(detected_red_flags),
            "legitimacy_count": len(detected_legitimacy),
        }

    # ------------------------------------------------------------------
    # Fast Keyword Saliency (<200ms Batched Inference)
    # ------------------------------------------------------------------
    def explain_fast(self, text: str, top_k: int = config.FAST_SALIENCY_TOP_K) -> Dict:
        """Sub-second explanation using batched token-masking perturbation."""
        words = re.findall(r"\b[A-Za-z0-9\$\@\%\!\?\.\,\'\-]+\b", text)
        if not words:
            return {"top_features": [], "html": None, "mode": "fast"}

        cleaned_base = clean_text(text)
        base_feat = self.extractor.extract_single(cleaned_base).reshape(1, -1)
        base_p_fake = float(self.model.predict_proba(base_feat)[0][1])

        # Filter out common stop words to focus perturbation on high-signal content
        stop_words = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
            "is", "are", "was", "were", "be", "this", "that", "it", "as", "by", "from",
        }

        unique_tokens = []
        seen = set()
        for w in words:
            w_lower = w.lower()
            if len(w) > 1 and w_lower not in stop_words and w_lower not in seen:
                seen.add(w_lower)
                unique_tokens.append(w)

        if not unique_tokens:
            unique_tokens = list(set(words))[:20]

        # Limit tokens to top 24 for ultra-fast single batch forward pass
        candidate_tokens = unique_tokens[:24]

        # Build masked variations
        masked_texts = []
        for token in candidate_tokens:
            pattern = re.compile(rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
            masked_str = pattern.sub("", cleaned_base)
            masked_texts.append(masked_str)

        # Batch forward pass
        batch_feats = self.extractor.extract_features(masked_texts, verbose=False)
        masked_probas = self.model.predict_proba(batch_feats)[:, 1]

        # Delta: if removing the token drops P(Fake), token had positive contribution toward Fake
        attributions = []
        for token, masked_p in zip(candidate_tokens, masked_probas):
            delta = float(base_p_fake - masked_p)
            attributions.append({"token": token, "shap_value": delta})

        ranked = sorted(attributions, key=lambda x: abs(x["shap_value"]), reverse=True)[:top_k]

        html_snippet = self._render_highlighted_html(text, ranked)

        return {
            "top_features": ranked,
            "html": html_snippet,
            "mode": "fast_saliency",
        }

    # ------------------------------------------------------------------
    # Deep SHAP Text Explainer (Partition Algorithm)
    # ------------------------------------------------------------------
    def _get_shap_explainer(self):
        if self._explainer is None:
            masker = shap.maskers.Text(r"\W+")
            self._explainer = shap.Explainer(
                self._predict_proba_from_raw_texts,
                masker,
                output_names=["Real", "Fake"],
                algorithm="partition",
            )
        return self._explainer

    def explain_shap(
        self,
        text: str,
        max_evals: int = config.SHAP_MAX_EVALS,
        top_k: int = config.SHAP_TOP_K,
    ) -> Dict:
        """Runs rigorous SHAP partition explainer over word perturbations."""
        explainer = self._get_shap_explainer()
        shap_values = explainer([text], max_evals=max_evals)
        fake_explanation = shap_values[0, :, "Fake"]

        tokens = list(fake_explanation.data)
        values = [float(v) for v in fake_explanation.values]

        ranked = sorted(zip(tokens, values), key=lambda pair: abs(pair[1]), reverse=True)
        top_features = [
            {"token": tok.strip(), "shap_value": val}
            for tok, val in ranked
            if tok.strip() and len(tok.strip()) > 1
        ][:top_k]

        try:
            html_content = shap.plots.text(fake_explanation, display=False)
        except Exception as exc:
            logger.warning("Could not render SHAP text plot HTML: %s", exc)
            html_content = self._render_highlighted_html(text, top_features)

        return {
            "top_features": top_features,
            "html": html_content,
            "mode": "shap_partition",
        }

    # ------------------------------------------------------------------
    # HTML Highlighter Helper
    # ------------------------------------------------------------------
    def _render_highlighted_html(self, raw_text: str, ranked_features: List[Dict]) -> str:
        """Builds a styled visual representation of highlighted trigger tokens."""
        feat_map = {f["token"].lower(): f["shap_value"] for f in ranked_features if abs(f["shap_value"]) > 0.001}
        if not feat_map:
            return f"<div style='font-family:sans-serif; color:#cbd5e1; line-height:1.6;'>{html.escape(raw_text)}</div>"

        max_val = max(abs(v) for v in feat_map.values()) or 1.0

        words = re.split(r"(\s+|[^\w\s])", raw_text)
        rendered_pieces = []
        for piece in words:
            p_lower = piece.lower().strip()
            if p_lower in feat_map:
                val = feat_map[p_lower]
                intensity = min(abs(val) / max_val, 1.0)
                if val > 0:  # Pushes toward Fake
                    bg = f"rgba(239, 68, 68, {0.2 + 0.65 * intensity:.2f})"
                    border = "#ef4444"
                    title = f"Pushes toward FAKE (+{val:.3f})"
                else:  # Pushes toward Real
                    bg = f"rgba(16, 185, 129, {0.2 + 0.65 * intensity:.2f})"
                    border = "#10b981"
                    title = f"Pushes toward REAL ({val:.3f})"

                span = (
                    f"<span style='background-color:{bg}; border-bottom: 2px solid {border}; "
                    f"padding: 2px 5px; border-radius: 4px; font-weight: 600; color: #ffffff;' "
                    f"title='{title}'>{html.escape(piece)}</span>"
                )
                rendered_pieces.append(span)
            else:
                rendered_pieces.append(html.escape(piece))

        full_html = (
            "<div style='font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; "
            "background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; "
            "padding: 16px; color: #e2e8f0; line-height: 1.8; font-size: 14px; white-space: pre-wrap;'>"
            + "".join(rendered_pieces)
            + "</div>"
        )
        return full_html

    # ------------------------------------------------------------------
    # Unified Prediction & Explanation Pipeline
    # ------------------------------------------------------------------
    def analyze(self, text: str, mode: str = "fast", max_evals: int = config.SHAP_MAX_EVALS) -> Dict:
        """Full end-to-end analysis: prediction, risk signals, and chosen explainability."""
        prediction = self.predict(text)
        risk_audit = self.audit_risk_signals(text)

        if mode == "shap":
            explanation = self.explain_shap(text, max_evals=max_evals)
        else:
            explanation = self.explain_fast(text)

        return {
            **prediction,
            "risk_audit": risk_audit,
            "explanation": explanation,
        }


@lru_cache(maxsize=1)
def get_detector() -> "FakeJobDetector":
    """Process-wide singleton so RoBERTa and XGBoost load once into memory."""
    return FakeJobDetector()


if __name__ == "__main__":
    detector = get_detector()
    sample_text = (
        "URGENT!!! Work from home and earn $5000 per week with no experience needed! "
        "Send your bank account details and wire $50 registration fee to get started. Contact HR on WhatsApp."
    )
    res = detector.analyze(sample_text, mode="fast")
    print("Verdict:", res["label"], "| Confidence:", round(res["confidence"], 3))
    print("Risk Tier:", res["risk_tier"])
    print("Red Flags:", len(res["risk_audit"]["red_flags"]))
    print("Top Influential Words:", [f["token"] for f in res["explanation"]["top_features"][:5]])
