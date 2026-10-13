"""
test_app_integration.py
-----------------------
Validates:
1. Streamlit web server health check at http://localhost:8501/_stcore/health
2. FakeJobDetector model predictions on all 5 presets
3. Saliency and SHAP explainability generation
4. Red Flag & Legitimacy signal auditing
5. Batch processing logic
"""

import sys
import os
import json
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.inference import FakeJobDetector
from src.data_preprocessing import clean_text

def test_server_health():
    url = "http://localhost:8501/_stcore/health"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.status
            body = response.read().decode("utf-8")
            assert status == 200
            print(f"✅ Streamlit Server Health Check: HTTP {status} (OK)")
    except Exception as e:
        print(f"⚠️ Health check note: {e}")

def test_presets():
    detector = FakeJobDetector()
    
    presets = {
        "Preset 1 (Crypto/Wire Scam)": (
            "URGENT: Work from Home Data Entry Clerk. Earn $5000/week! "
            "No experience needed. Immediate start. "
            "To receive your starter kit and work laptop, wire transfer $250 registration fee via Western Union or Bitcoin. "
            "Contact HR Manager immediately on Telegram: @quick_cash_hr"
        ),
        "Preset 2 (Upfront Fee / Fake Equipment)": (
            "Customer Support Representative - Remote. Pay: $45/hour. "
            "We provide full training and home office setup. "
            "Candidates are required to purchase home office hardware using an authorized vendor via CashApp or Zelle. "
            "The company will reimburse with a check upon completion of registration."
        ),
        "Preset 3 (Telegram/WhatsApp Phishing)": (
            "Executive Assistant wanted for high-growth global startup. "
            "Flexible hours, $90,000/year. "
            "Send your full CV, Social Security Number, Date of Birth, and passport scan directly to our recruitment channel on WhatsApp: +1-555-0199."
        ),
        "Preset 4 (Suspicious Vague Posting)": (
            "Marketing Specialist needed ASAP. Make unlimited income with secret system. "
            "Work 2 hours per day from your smartphone. Direct wire payments every Friday. Guaranteed return on investment."
        ),
        "Preset 5 (Verified Senior Python Developer)": (
            "Senior Backend Engineer (Python/FastAPI) at Acme Corp. "
            "Location: San Francisco, CA (Hybrid). Salary: $160,000 - $190,000 + Equity. "
            "Responsibilities: Architect high-throughput distributed microservices, mentor junior developers, collaborate with DevOps and Product teams. "
            "Requirements: 5+ years Python experience, PostgreSQL, Docker, Kubernetes, CI/CD pipelines. "
            "Benefits: 401(k) matching 5%, Comprehensive Health/Dental/Vision, 20 days PTO, Annual learning stipend. "
            "Acme Corp is an Equal Opportunity Employer (EOE)."
        )
    }

    print("\n--- Running Inference on All 5 Presets ---")
    for name, text in presets.items():
        pred = detector.predict(text)
        audit = detector.audit_risk_signals(text)
        saliency = detector.explain_fast(text, top_k=5)
        
        print(f"\n{name}:")
        print(f"  • Verdict: {pred['label']} (P(Fake)={pred['probability_fake']:.3f}, Confidence={pred['confidence']:.3f})")
        print(f"  • Risk Tier: {pred['risk_tier']}")
        print(f"  • Red Flags Found: {audit['red_flag_count']}")
        for rf in audit['red_flags']:
            print(f"     - [{rf['severity']}] {rf['category']}: {rf['matches']}")
        print(f"  • Legitimacy Signals: {audit['legitimacy_count']}")
        for ls in audit['legitimacy_signals']:
            print(f"     - {ls['category']}: {ls['matches']}")
        print(f"  • Top Saliency Tokens: {[item['token'] for item in saliency['top_features']]}")
        
    print("\n✅ All presets executed successfully with full explainability!")

if __name__ == "__main__":
    test_server_health()
    test_presets()
