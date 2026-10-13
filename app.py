"""
app.py
------
Fake Job Posting Detector - Interactive Web Application.
Features:
  1. Real-Time Job Scanner with Dual-Mode Explainability (Fast Saliency & Deep SHAP)
  2. Rule-Based Red Flag & Legitimacy Auditor
  3. Batch File Analyzer (CSV/JSON upload with export)
  4. Model Intelligence & Performance Analytics Dashboard
  5. Interactive Risk Sandbox & "What-If" Trigger Simulator
  6. Session-based Authentication & Audit History
"""

import html
import io
import json
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src import auth, config
from src.inference import get_detector

# ---------------------------------------------------------------------------
# Page Configuration & Custom CSS Design System
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake Job Posting Detector | AI Fraud Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global Font & Theme */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main Container Background */
.stApp {
    background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 0.95), rgba(2, 6, 23, 1));
    color: #f8fafc;
}

/* Glassmorphism Card Style */
.glass-card {
    background: rgba(30, 41, 59, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
}

/* Header Gradient */
.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 24px;
    line-height: 1.5;
}

/* Metric Highlight Boxes */
.metric-box {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: all 0.2s ease;
}

.metric-box:hover {
    background: rgba(30, 41, 59, 0.85);
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f8fafc;
}

.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Status Badges */
.badge-fake {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.3) 100%);
    border: 1px solid #ef4444;
    color: #fca5a5;
    padding: 8px 16px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    letter-spacing: 0.03em;
}

.badge-real {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
    border: 1px solid #10b981;
    color: #6ee7b7;
    padding: 8px 16px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    letter-spacing: 0.03em;
}

.badge-caution {
    background: linear-gradient(135deg, rgba(234, 179, 8, 0.2) 0%, rgba(202, 138, 4, 0.3) 100%);
    border: 1px solid #eab308;
    color: #fde047;
    padding: 8px 16px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
}

/* Severity Pills */
.pill-critical {
    background: #ef4444;
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    text-transform: uppercase;
}

.pill-high {
    background: #f97316;
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    text-transform: uppercase;
}

.pill-medium {
    background: #eab308;
    color: #000000;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
    text-transform: uppercase;
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background-color: #0b0f19;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Primary Button Styling */
button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def _init_session():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "scanner_text_area" not in st.session_state:
        st.session_state.scanner_text_area = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def _logout():
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()


# ---------------------------------------------------------------------------
# Model Singleton Resource
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="⚡ Initializing RoBERTa Transformer & XGBoost Engine...")
def load_detector_resource():
    return get_detector()


# ---------------------------------------------------------------------------
# Presets & Sample Library
# ---------------------------------------------------------------------------
SAMPLE_PRESETS = {
    "🚨 Wire Transfer / Fee Scam": (
        "URGENT HIRING - Online Data Entry Clerk Needed ASAP! Make $4500/week guaranteed with NO EXPERIENCE! "
        "Work from the comfort of your home 2 hours a day. All applicants accepted immediately. "
        "To receive your starter kit, welcome package, and check for office supplies, send your bank account details "
        "and wire a refundable registration fee of $50 via CashApp, Zelle, or Western Union to our HR account. "
        "Contact HR Manager immediately on WhatsApp: +1-888-555-SCAM."
    ),
    "🚨 WhatsApp Crypto Assistant": (
        "Cryptocurrency Payment Processing Associate ($80/hr - Remote). "
        "We are looking for immediate candidates to assist with processing customer crypto transactions and "
        "inspecting forwarded luxury packages. 100% guaranteed income, no interview required. "
        "Must have an active personal bank account to receive company deposits and buy Bitcoin vouchers. "
        "Apply right now via Telegram @instant_crypto_hire or email direct to job-desk-hiring@gmail.com. ACT NOW!"
    ),
    "🚨 Mystery Shopper Mule Scheme": (
        "Mystery Shopper / Quality Assurance Inspector Wanted! Earn $500 cash per assignment. "
        "We will send you cashier's checks. Deposit the check into your personal bank account, keep $500 for your fee, "
        "and wire back the excess funds through MoneyGram. Fast money, limited spots available today!"
    ),
    "🛡️ Legit Senior Backend Engineer": (
        "Senior Software Engineer (Distributed Systems) at Stripe. We are looking for an experienced backend engineer "
        "to join our Core Infrastructure team. Responsibilities: Design and maintain high-throughput transaction processing "
        "services, improve database reliability, and collaborate on API design. Requirements: 5+ years of software engineering "
        "experience with Python or Go, strong familiarity with SQL and distributed systems, Bachelor's degree in Computer Science "
        "or equivalent practical experience. We offer comprehensive healthcare (medical, dental, vision), 401(k) retirement "
        "matching up to 6%, flexible PTO, annual learning stipend, and competitive equity."
    ),
    "🛡️ Legit Clinical Healthcare RN": (
        "Staff Registered Nurse (ICU) - Cleveland Clinic. We are seeking a compassionate, licensed RN for our Intensive Care Unit. "
        "Requirements: Active state Registered Nurse license, current BLS and ACLS certification, minimum 2 years acute care "
        "or critical care experience. Responsibilities include patient monitoring, administering medications, and collaborating with "
        "multidisciplinary care teams. Full-time role with shift differential, medical/dental/life insurance, pension plan, "
        "and tuition reimbursement."
    ),
}


# ---------------------------------------------------------------------------
# Authentication Pages
# ---------------------------------------------------------------------------
def render_login_view():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; margin-top: 40px;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 3.5rem;'>🛡️</div>", unsafe_allow_html=True)
        st.markdown("<div class='hero-title'>Fake Job Posting Detector</div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color: #94a3b8;'>Enterprise Explainable AI for Employment Fraud Protection</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔐 Sign In", "✨ Create Account"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="e.g. demo or admin")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("Sign In to Portal", type="primary", use_container_width=True)

            if submit_login:
                user = auth.authenticate(username.strip(), password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.toast(f"Welcome back, {user['display_name']}!", icon="👋")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")

            st.markdown("<div style='margin-top: 15px; font-size: 0.85rem; color: #94a3b8;'>", unsafe_allow_html=True)
            st.markdown("<strong>Demo Credentials:</strong>")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if st.button("🔑 Fill Demo User", use_container_width=True):
                    user = auth.authenticate("demo", "demo1234")
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.rerun()
            with col_d2:
                if st.button("👑 Fill Admin", use_container_width=True):
                    user = auth.authenticate("admin", "admin123")
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_register:
            with st.form("register_form"):
                new_user = st.text_input("New Username", placeholder="Enter username")
                new_name = st.text_input("Display Name", placeholder="e.g. Alex Chen")
                new_pass = st.text_input("New Password", type="password", placeholder="Choose secure password")
                submit_reg = st.form_submit_button("Create Account", use_container_width=True)

            if submit_reg:
                if not new_user or not new_pass:
                    st.warning("Please fill in both username and password.")
                else:
                    ok = auth.register_user(new_user.strip(), new_pass, new_name.strip() or None)
                    if ok:
                        st.success("Account created successfully! You can now log in.")
                    else:
                        st.error("Username already exists. Please choose a different one.")

        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------
def render_main_dashboard():
    detector = load_detector_resource()
    user = st.session_state.user

    # Sidebar Navigation & User Info
    with st.sidebar:
        st.markdown(
            f"""
            <div style='background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.1); 
                        border-radius: 12px; padding: 14px; margin-bottom: 20px;'>
                <div style='font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;'>Logged in as</div>
                <div style='font-size: 1.1rem; font-weight: 700; color: #f8fafc;'>{user['display_name']}</div>
                <div style='font-size: 0.8rem; color: #6366f1;'>@{user['username']} • Security Analyst</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### ⚙️ Engine Settings")
        scan_mode = st.radio(
            "Explainability Mode",
            options=["⚡ Fast Saliency (<200ms)", "🔬 Deep SHAP Partition"],
            help="Fast Saliency gives instant sub-second word importance via batched perturbations. Deep SHAP computes full game-theoretic Shapley values.",
        )
        mode_key = "shap" if "Deep SHAP" in scan_mode else "fast"

        shap_evals = 120
        if mode_key == "shap":
            shap_evals = st.slider(
                "SHAP Thoroughness (evals)",
                min_value=40,
                max_value=300,
                value=config.SHAP_MAX_EVALS,
                step=20,
                help="Higher evals increase attribution precision at the expense of evaluation time.",
            )

        st.divider()

        # Quick Stats in Sidebar
        scans_count = len(st.session_state.history)
        flagged_count = sum(1 for item in st.session_state.history if item.get("label") == "Fake")
        st.markdown("### 📈 Session Summary")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Total Scans", scans_count)
        with col_s2:
            st.metric("Flagged Fake", flagged_count)

        st.divider()
        if st.button("🚪 Sign Out", use_container_width=True):
            _logout()

    # Main Navigation Tabs: Live Job Scanner and Scan History
    tab_scanner, tab_history = st.tabs([
        "🔍 Live Job Scanner",
        "📜 Scan History",
    ])

    # -----------------------------------------------------------------------
    # TAB 1: Live Job Scanner
    # -----------------------------------------------------------------------
    with tab_scanner:
        st.markdown("<div class='hero-title'>Job Fraud Intelligence Scanner</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hero-subtitle'>Paste any job posting text or load a scenario to detect fraudulent openings, "
            "highlight manipulative phrases, and audit security red flags.</div>",
            unsafe_allow_html=True,
        )

        def _set_preset_text(sample_text: str):
            st.session_state.scanner_text_area = sample_text
            st.session_state.last_result = None

        def _clear_input_text():
            st.session_state.scanner_text_area = ""
            st.session_state.last_result = None

        # Preset Buttons Row
        st.markdown("**⚡ Quick Load Presets (Click any to test):**")
        preset_cols = st.columns(len(SAMPLE_PRESETS))
        for i, (name, sample_text) in enumerate(SAMPLE_PRESETS.items()):
            with preset_cols[i]:
                st.button(
                    name,
                    key=f"btn_preset_{i}",
                    on_click=_set_preset_text,
                    args=(sample_text,),
                    use_container_width=True,
                )

        # Text Area Input
        text_input = st.text_area(
            "Job Posting Content",
            height=200,
            placeholder="Paste complete job details: Title, Company Profile, Description, Requirements, Benefits, Contact info...",
            key="scanner_text_area",
        )

        col_act1, col_act2, col_act3 = st.columns([2, 1, 1])
        with col_act1:
            btn_scan = st.button("🛡️ Scan & Analyze Job Posting", type="primary", use_container_width=True)
        with col_act2:
            st.button("🗑️ Clear Input", on_click=_clear_input_text, use_container_width=True)

        # Handle Scan Action
        if btn_scan:
            if not text_input or len(text_input.strip()) < 15:
                st.warning("⚠️ Please provide a sufficiently descriptive job posting (at least 15 characters).")
            else:
                with st.spinner("Analyzing linguistic patterns, transformer semantics, and fraud indicators..."):
                    start_t = time.time()
                    
                    # Unified end-to-end analysis
                    result_payload = detector.analyze(text_input, mode=mode_key, max_evals=shap_evals)
                    
                    latency = round(time.time() - start_t, 2)
                    result_payload["latency"] = latency
                    result_payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    result_payload["snippet"] = text_input[:70] + "..." if len(text_input) > 70 else text_input
                    result_payload["text"] = text_input

                    # Cache in session
                    st.session_state.last_result = result_payload
                    st.session_state.history.insert(0, result_payload)

        # Render Results View
        if st.session_state.last_result:
            render_scan_results(st.session_state.last_result)

    # -----------------------------------------------------------------------
    # TAB 2: Scan History Log
    # -----------------------------------------------------------------------
    with tab_history:
        render_history_log()


# ---------------------------------------------------------------------------
# Render Scan Results
# ---------------------------------------------------------------------------
def render_scan_results(res: dict):
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)

    verdict_label = res.get("label", "Real")
    risk_tier = res.get("risk_tier", "Low Risk")
    risk_badge = res.get("risk_badge", "🟢 VERIFIED LEGITIMATE")
    confidence = res.get("confidence", 0.95)
    prob_fake = res.get("probability_fake", 0.05)
    prob_real = res.get("probability_real", 0.95)
    risk_color = res.get("risk_color", "#10b981")
    latency = res.get("latency", 0.0)

    banner_bg = "rgba(239, 68, 68, 0.15)" if verdict_label == "Fake" else "rgba(16, 185, 129, 0.15)"
    banner_border = risk_color

    st.markdown(
        f"""
        <div style='background: {banner_bg}; border: 1px solid {banner_border}; border-radius: 16px; 
                    padding: 24px; margin-bottom: 25px; display: flex; flex-wrap: wrap; 
                    align-items: center; justify-content: space-between;'>
            <div>
                <span class='badge-tier' style='background: {risk_color}22; color: {risk_color}; border: 1px solid {risk_color}66;'>
                    {risk_badge}
                </span>
                <div style='font-size: 1.8rem; font-weight: 800; color: #ffffff; margin-top: 8px;'>
                    Predicted: <span style='color: {risk_color};'>{verdict_label.upper()} POSTING</span>
                </div>
                <div style='font-size: 0.95rem; color: #cbd5e1; margin-top: 4px;'>
                    Classification Confidence: <strong>{confidence*100:.1f}%</strong> • Latency: <strong>{latency}s</strong>
                </div>
            </div>
            <div style='display: flex; gap: 15px; margin-top: 10px;'>
                <div style='background: rgba(15, 23, 42, 0.8); padding: 12px 20px; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.08);'>
                    <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;'>P(Fraudulent)</div>
                    <div style='font-size: 1.4rem; font-weight: 700; color: #ef4444;'>{prob_fake*100:.1f}%</div>
                </div>
                <div style='background: rgba(15, 23, 42, 0.8); padding: 12px 20px; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.08);'>
                    <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;'>P(Legitimate)</div>
                    <div style='font-size: 1.4rem; font-weight: 700; color: #10b981;'>{prob_real*100:.1f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Breakdown Columns: Audit Signals & Attribution
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 🔎 Detected Risk & Legitimacy Signals")
        risk_audit = res.get("risk_audit", {})
        red_flags = risk_audit.get("red_flags", [])
        legitimacy = risk_audit.get("legitimacy_signals", [])

        if red_flags:
            st.markdown(f"**🚨 High-Risk Fraud Triggers Found ({len(red_flags)}):**")
            for flag in red_flags:
                pill_class = "pill-critical" if flag["severity"] == "CRITICAL" else ("pill-high" if flag["severity"] == "HIGH" else "pill-medium")
                st.markdown(
                    f"""
                    <div style='background: rgba(239, 68, 68, 0.08); border-left: 3px solid #ef4444; 
                                padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <strong style='color: #fca5a5;'>{flag["category"]}</strong>
                            <span class='{pill_class}'>{flag["severity"]}</span>
                        </div>
                        <div style='font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;'>{flag["description"]}</div>
                        <div style='font-size: 0.8rem; color: #f87171; margin-top: 4px;'>
                            Matched terms: <code>{", ".join(flag["matches"])}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ No explicit high-risk scam triggers or upfront payment requests detected.")

        if legitimacy:
            st.markdown(f"**🛡️ Verified Corporate Signals ({len(legitimacy)}):**")
            for leg in legitimacy:
                st.markdown(
                    f"""
                    <div style='background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; 
                                padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                        <strong style='color: #6ee7b7;'>{leg["category"]}</strong>
                        <div style='font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;'>{leg["description"]}</div>
                        <div style='font-size: 0.8rem; color: #34d399; margin-top: 4px;'>
                            Matched terms: <code>{", ".join(leg["matches"])}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Safety Recommendations
        with st.expander("🛡️ Applicant Safety Recommendations"):
            st.markdown(
                """
                - **Never pay for starter kits, background checks, or equipment upfront.** Legitimate employers provide necessary equipment at no cost.
                - **Beware of checks sent with instructions to wire money back.** This is always a counterfeit check scam.
                - **Verify interview channels.** Legitimate companies conduct interviews via official corporate email domains, Microsoft Teams, Zoom, or formal in-person visits.
                - **Cross-check the job on the company's official careers page.**
                """
            )

    with col_right:
        st.markdown("### 📊 Word-Level Attribution")
        explanation = res.get("explanation", {})
        top_features = explanation.get("top_features", [])

        if top_features:
            df_feat = pd.DataFrame(top_features)
            df_feat["Impact"] = df_feat["shap_value"].apply(lambda v: "Pushes Fake" if v > 0 else "Pushes Real")
            df_feat["Color"] = df_feat["shap_value"].apply(lambda v: "#ef4444" if v > 0 else "#10b981")
            df_feat = df_feat.sort_values(by="shap_value", ascending=True)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    y=df_feat["token"],
                    x=df_feat["shap_value"],
                    orientation="h",
                    marker_color=df_feat["Color"],
                    hovertemplate="<b>Word:</b> %{y}<br><b>Impact:</b> %{x:.4f}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Top Words Driving the Prediction (Red = Fake, Green = Real)",
                xaxis_title="Contribution Delta toward P(Fake)",
                yaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
                margin=dict(l=10, r=10, t=40, b=20),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No single high-impact trigger words exceeded attribution thresholds.")

        html_view = explanation.get("html")
        if html_view:
            st.markdown("**🎨 Visual Word Highlighter:**")
            components.html(html_view, height=220, scrolling=True)


# ---------------------------------------------------------------------------
# Scan History Log
# ---------------------------------------------------------------------------
def render_history_log():
    st.markdown("<div class='hero-title'>Session Scan History</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Review past job postings analyzed during this browser session.</div>",
        unsafe_allow_html=True,
    )

    history = st.session_state.history
    if not history:
        st.info("No job postings scanned in this session yet. Run a scan in the Live Scanner tab.")
        return

    hist_rows = []
    for item in history:
        hist_rows.append({
            "Timestamp": item.get("timestamp"),
            "Snippet": item.get("snippet"),
            "Verdict": item.get("label"),
            "Confidence": f"{item.get('confidence', 0)*100:.1f}%",
            "Risk Tier": item.get("risk_tier"),
            "P(Fake)": round(item.get("probability_fake", 0), 3),
            "Latency (s)": item.get("latency", 0),
        })

    df_hist = pd.DataFrame(hist_rows)
    st.dataframe(df_hist, use_container_width=True)


# ---------------------------------------------------------------------------
# Application Entrypoint
# ---------------------------------------------------------------------------
def main():
    _init_session()
    if not st.session_state.authenticated:
        render_login_view()
    else:
        render_main_dashboard()


if __name__ == "__main__":
    main()
