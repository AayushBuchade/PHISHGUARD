"""
PHISHGUARD — AI-Powered Email Threat Detection
Streamlit Cyber SOC Application (Dark Cybersecurity Console)
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Add workspace directory to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.security_analyzer import SecurityAnalyzer
from src.feature_engineering import extract_cyber_features
from src.explainability import EmailExplainer
from src.report_generator import generate_forensic_pdf
from src.text_preprocessor import clean_text

# Configure page layout and cybersecurity dark SOC theme
st.set_page_config(
    page_title="PHISHGUARD // Cyber Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Cyber SOC Dark Mode CSS
st.markdown("""
<style>
    /* Global Reset & Cyber Dark Palette */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #0b0f19 !important;
        color: #e6edf3 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* Header Bar */
    header[data-testid="stHeader"] {
        background-color: #0b0f19 !important;
        border-bottom: 1px solid #21262d !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #21262d !important;
    }
    section[data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #f0f6fc !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f6fc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    p, span, label {
        color: #c9d1d9 !important;
    }
    
    /* Top Status Bar */
    .soc-topbar {
        background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        flex-wrap: wrap;
        gap: 12px;
    }
    .soc-topbar-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .soc-topbar-pills {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pill-green {
        background: rgba(63, 185, 80, 0.15);
        color: #3fb950;
        border: 1px solid rgba(63, 185, 80, 0.4);
    }
    .pill-blue {
        background: rgba(88, 166, 255, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(88, 166, 255, 0.4);
    }
    .pill-red {
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.4);
    }
    
    /* Threat Cards */
    .soc-card {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    .soc-card-header {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e !important;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Metrics */
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 4px;
        font-family: 'Consolas', 'Courier New', monospace;
    }
    
    /* Custom High-Contrast Textarea */
    div[data-testid="stTextArea"] textarea {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-family: 'Consolas', 'Courier New', monospace !important;
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
        padding: 14px !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.25) !important;
    }
    
    /* Custom Input Labels & Text */
    div[data-testid="stTextArea"] label, div[data-testid="stRadio"] label, div[data-testid="stFileUploader"] label {
        color: #f0f6fc !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* Radio Option Text Color */
    div[data-testid="stRadio"] div[role="radiogroup"] label div p {
        color: #e6edf3 !important;
        font-weight: 500 !important;
    }

    /* Primary Buttons (Analyze) */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #e5534b 0%, #d73a49 100%) !important;
        color: #ffffff !important;
        border: 1px solid #f85149 !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 14px rgba(215, 58, 73, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(215, 58, 73, 0.6) !important;
    }

    /* Secondary / Sample Buttons */
    button[kind="secondary"], .stButton > button {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover, .stButton > button:hover {
        background-color: #21262d !important;
        border-color: #58a6ff !important;
        color: #58a6ff !important;
    }
    
    /* Badges */
    .badge-critical {
        background-color: rgba(248, 81, 73, 0.2);
        color: #ff7b72 !important;
        border: 1px solid #f85149;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        letter-spacing: 0.04em;
    }
    .badge-suspicious {
        background-color: rgba(210, 153, 34, 0.2);
        color: #f2cc60 !important;
        border: 1px solid #d29922;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        letter-spacing: 0.04em;
    }
    .badge-clean {
        background-color: rgba(63, 185, 80, 0.2);
        color: #56d364 !important;
        border: 1px solid #3fb950;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        letter-spacing: 0.04em;
    }
    
    /* Dataframe Table Dark Theme */
    div[data-testid="stDataFrame"] {
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        background-color: #161b22 !important;
    }

    /* Defanged Code Blocks */
    .defanged-code {
        font-family: 'Consolas', 'Courier New', monospace;
        background-color: #0d1117;
        color: #79c0ff !important;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.88rem;
        border: 1px solid #30363d;
        word-break: break-all;
    }
    
    /* Risk Progress Bar */
    .risk-bar-container {
        width: 100%;
        background-color: #21262d;
        border-radius: 6px;
        height: 14px;
        overflow: hidden;
        margin: 8px 0;
        border: 1px solid #30363d;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }
</style>
""", unsafe_allow_html=True)


# Lazy loading singletons for ML models
@st.cache_resource
def load_ml_pipeline():
    models_dir = os.path.join(BASE_DIR, "models")
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    all_models_path = os.path.join(models_dir, "all_models.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")
    comp_path = os.path.join(models_dir, "model_comparison.json")
    feat_path = os.path.join(models_dir, "feature_importance.json")
    
    best_model = joblib.load(best_model_path)
    vectorizer = joblib.load(vec_path)
    all_models = joblib.load(all_models_path)
    
    with open(meta_path) as f:
        metadata = json.load(f)
    with open(comp_path) as f:
        comparison = json.load(f)
    with open(feat_path) as f:
        feature_imp = json.load(f)
        
    explainer = EmailExplainer(model=best_model, vectorizer=vectorizer)
    analyzer = SecurityAnalyzer()
    
    return {
        "best_model": best_model,
        "vectorizer": vectorizer,
        "all_models": all_models,
        "metadata": metadata,
        "comparison": comparison,
        "feature_importance": feature_imp,
        "explainer": explainer,
        "analyzer": analyzer
    }


pipeline = load_ml_pipeline()
analyzer = pipeline["analyzer"]
explainer = pipeline["explainer"]

# Initialize Session State
if "history" not in st.session_state:
    st.session_state["history"] = []
if "current_analysis" not in st.session_state:
    st.session_state["current_analysis"] = None
if "email_text_input" not in st.session_state:
    st.session_state["email_text_input"] = ""


# Core Analysis Function
def analyze_email_payload(raw_content: str, source_type: str = "Pasted Text"):
    parsed = analyzer.parse_eml_content(raw_content)
    body_text = parsed["body_text"]
    
    iocs = analyzer.extract_iocs(body_text)
    sender_analysis = analyzer.check_sender_mismatch(
        parsed["from"], parsed["return_path"], parsed["reply_to"]
    )
    auth_analysis = analyzer.parse_auth_headers(parsed["headers"])
    cyber_features = extract_cyber_features(body_text)
    
    cleaned_input = clean_text(body_text)
    tfidf_vec = pipeline["vectorizer"].transform([cleaned_input])
    
    probs = pipeline["best_model"].predict_proba(tfidf_vec)[0]
    phish_prob = float(probs[1])
    
    model_preds = {}
    for m_name, m_obj in pipeline["all_models"].items():
        m_prob = float(m_obj.predict_proba(tfidf_vec)[0][1])
        model_preds[m_name] = {
            "phishing_prob": m_prob,
            "prediction": "Phishing" if m_prob >= 0.5 else "Legitimate"
        }
        
    xai = explainer.explain_instance_weights(body_text, top_k=10)
    
    risk = analyzer.calculate_risk_score(
        ml_probability=phish_prob,
        cyber_features=cyber_features,
        iocs=iocs,
        sender_analysis=sender_analysis,
        auth_analysis=auth_analysis,
        attachments=parsed["attachments"]
    )
    
    snippets = analyzer.find_suspicious_snippets(body_text)
    case_id = f"PG-{time.strftime('%Y%m%d')}-{len(st.session_state['history']) + 1:03d}"
    
    analysis_record = {
        "case_id": case_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_type": source_type,
        "parsed_email": parsed,
        "cyber_features": cyber_features,
        "iocs": iocs,
        "sender_analysis": sender_analysis,
        "auth_analysis": auth_analysis,
        "ml_probability": phish_prob,
        "model_predictions": model_preds,
        "xai": xai,
        "risk_assessment": risk,
        "snippets": snippets,
        "raw_content": raw_content
    }
    
    st.session_state["current_analysis"] = analysis_record
    st.session_state["history"].insert(0, analysis_record)
    return analysis_record


# Sidebar Navigation & Threat Engine Status
with st.sidebar:
    st.markdown("<div style='display:flex; align-items:center; gap:10px; margin-bottom:4px;'><span style='font-size:1.8rem;'>🛡️</span><span style='font-size:1.35rem; font-weight:800; color:#f0f6fc;'>PHISHGUARD</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem; color:#8b949e; margin-bottom:18px; letter-spacing:0.05em;'>AI EMAIL THREAT DETECTION SYSTEM</div>", unsafe_allow_html=True)
    
    page = st.radio(
        "NAVIGATION",
        ["1. Email Analyzer", "2. Risk Assessment", "3. ML Dashboard", "4. Investigation Report"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<div class='soc-card-header'>ENGINE BENCHMARKS</div>", unsafe_allow_html=True)
    st.markdown(f"**Model:** `{pipeline['metadata']['best_model_name']}`")
    st.markdown(f"**Test F1-Score:** `{pipeline['metadata']['best_model_metrics']['F1-Score'] * 100:.2f}%`")
    st.markdown(f"**ROC-AUC:** `{pipeline['metadata']['best_model_metrics']['ROC-AUC']:.4f}`")
    st.markdown(f"**Corpus Size:** `{pipeline['metadata']['total_dataset_samples']:,} emails`")
    
    st.markdown("---")
    st.markdown("<div class='soc-card-header'>DEFENSIVE SAFETY</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem; color:#8b949e; line-height:1.4;'>🔒 <b>Strict Passive Inspection:</b> URLs & IPs are defanged (<code>hxxp</code>). No external network requests are made, and attachments are never executed.</div>", unsafe_allow_html=True)


# Top Cyber Status Header (Shown on all pages)
st.markdown(f"""
<div class="soc-topbar">
    <div class="soc-topbar-left">
        <span style="font-size: 1.25rem;">🛡️</span>
        <div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #f0f6fc; letter-spacing: -0.01em;">
                PHISHGUARD SECURITY CONSOLE // SOC TIER-2
            </div>
            <div style="font-size: 0.75rem; color: #8b949e;">
                AI-Powered Threat Engine • Sublinear TF-IDF • Calibrated Linear SVM
            </div>
        </div>
    </div>
    <div class="soc-topbar-pills">
        <span class="status-pill pill-green">● INFERENCE ONLINE</span>
        <span class="status-pill pill-blue">LATENCY &lt; 2MS</span>
        <span class="status-pill pill-green">5,000 N-GRAMS</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: EMAIL ANALYZER
# ==============================================================================
if page == "1. Email Analyzer":
    st.markdown("### 📥 Email Threat Analyzer")
    st.markdown("Paste suspicious email headers and body text, or load a pre-configured SOC threat scenario:")
    
    # Pre-loaded sample buttons that reliably write to session_state
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        if st.button("🚨 Load Phishing (PayPal)", use_container_width=True):
            sample_path = os.path.join(BASE_DIR, "samples", "phishing_account_suspended.eml")
            if os.path.exists(sample_path):
                with open(sample_path, "r", encoding="utf-8") as f:
                    st.session_state["email_text_input"] = f.read()
                st.rerun()
                
    with col_s2:
        if st.button("⚠️ Load CEO BEC Scam (Wire)", use_container_width=True):
            sample_path = os.path.join(BASE_DIR, "samples", "phishing_urgent_wire_transfer.eml")
            if os.path.exists(sample_path):
                with open(sample_path, "r", encoding="utf-8") as f:
                    st.session_state["email_text_input"] = f.read()
                st.rerun()
                
    with col_s3:
        if st.button("✅ Load Clean (Corporate)", use_container_width=True):
            sample_path = os.path.join(BASE_DIR, "samples", "legitimate_project_update.eml")
            if os.path.exists(sample_path):
                with open(sample_path, "r", encoding="utf-8") as f:
                    st.session_state["email_text_input"] = f.read()
                st.rerun()
                
    input_method = st.radio("Input Method:", ["Paste Email Content", "Upload .EML File"], horizontal=True)
    
    raw_input_email = ""
    source_name = "Pasted Text"
    
    if input_method == "Upload .EML File":
        uploaded_file = st.file_uploader("Select an .eml email file to upload:", type=["eml", "txt"])
        if uploaded_file is not None:
            raw_input_email = uploaded_file.getvalue().decode("utf-8", errors="replace")
            source_name = f"Uploaded File: {uploaded_file.name}"
            st.session_state["email_text_input"] = raw_input_email
    else:
        raw_input_email = st.text_area(
            "Email Headers and Body Content (RFC 822 format or plain text):",
            value=st.session_state["email_text_input"],
            height=260,
            placeholder="From: Security Team <security@bank-protection-alert.xyz>\nReply-To: credential-collector@drop-mail.pw\nSubject: URGENT: Your Account Has Been Suspended!\n\nDear Customer,\nWe detected unauthorized access to your account. Verify immediately at http://192.168.1.1/login or your account will be deleted within 24 hours...",
            key="email_text_input"
        )
        
    col_act1, col_act2 = st.columns([1, 3])
    with col_act1:
        analyze_clicked = st.button("⚡ DETECT THREAT & ANALYZE", type="primary", use_container_width=True)
    with col_act2:
        if st.button("Clear Input", use_container_width=False):
            st.session_state["email_text_input"] = ""
            st.session_state["current_analysis"] = None
            st.rerun()
            
    if analyze_clicked:
        current_text = st.session_state.get("email_text_input", "").strip()
        if not current_text:
            st.error("⚠️ Please paste email text or upload an .eml file before analyzing.")
        else:
            with st.spinner("Analyzing NLP features, extracting IOCs, and calculating threat risk score..."):
                analysis = analyze_email_payload(current_text, source_type=source_name)
                st.success(f"Analysis complete for Case ID: **{analysis['case_id']}**!")
                
    # If analysis exists, display immediate triage summary cards
    if st.session_state["current_analysis"] is not None:
        cur = st.session_state["current_analysis"]
        risk = cur["risk_assessment"]
        
        st.markdown("---")
        st.markdown("#### ⚡ Immediate Triage Summary")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            badge_class = "badge-critical" if "Phish" in risk["verdict"] else ("badge-suspicious" if "Suspicious" in risk["verdict"] else "badge-clean")
            st.markdown(f"""
            <div class='soc-card'>
                <div class='soc-card-header'>CLASSIFICATION VERDICT</div>
                <div class='{badge_class}' style='font-size:1.15rem; padding:6px 14px;'>{risk['verdict'].upper()}</div>
                <div style='font-size:0.75rem; color:#8b949e; margin-top:8px;'>Level: {risk['risk_level']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class='soc-card'>
                <div class='soc-card-header'>THREAT RISK SCORE</div>
                <div class='metric-val' style='color:{risk['badge_color']};'>{risk['score']}<span style='font-size:1.1rem; color:#8b949e;'> / 100</span></div>
                <div class='risk-bar-container'>
                    <div class='risk-bar-fill' style='width:{risk['score']}%; background-color:{risk['badge_color']};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class='soc-card'>
                <div class='soc-card-header'>ML PROBABILITY</div>
                <div class='metric-val' style='color:#58a6ff;'>{cur['ml_probability'] * 100:.1f}%</div>
                <div style='font-size:0.75rem; color:#8b949e; margin-top:8px;'>Model: Linear SVM (Calibrated)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
            total_iocs = cur['iocs']['total_urls'] + len(cur['iocs']['ip_addresses'])
            st.markdown(f"""
            <div class='soc-card'>
                <div class='soc-card-header'>EXTRACTED IOCs</div>
                <div class='metric-val' style='color:#f0f6fc;'>{total_iocs}</div>
                <div style='font-size:0.75rem; color:#8b949e; margin-top:8px;'>URLs: {cur['iocs']['total_urls']} | IPs: {len(cur['iocs']['ip_addresses'])}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.info("👉 Switch to **2. Risk Assessment** in the sidebar for complete factor breakdowns and highlighted keywords, or **4. Investigation Report** to export a PDF.")


# ==============================================================================
# PAGE 2: RISK ASSESSMENT
# ==============================================================================
elif page == "2. Risk Assessment":
    st.markdown("### 📊 Comprehensive Threat Risk Assessment")
    
    if st.session_state["current_analysis"] is None:
        st.warning("⚠️ No email analyzed yet. Please submit an email in **1. Email Analyzer** first.")
    else:
        cur = st.session_state["current_analysis"]
        risk = cur["risk_assessment"]
        parsed = cur["parsed_email"]
        
        st.markdown(f"**Case Reference:** `{cur['case_id']}` | **Subject:** `{parsed.get('subject', 'N/A')}`")
        
        # Top KPI Metric Banners
        c_verdict, c_score, c_ml, c_auth = st.columns(4)
        with c_verdict:
            badge_class = "badge-critical" if "Phish" in risk["verdict"] else ("badge-suspicious" if "Suspicious" in risk["verdict"] else "badge-clean")
            st.markdown(f"""
            <div class='soc-card'>
                <div class='soc-card-header'>CLASSIFICATION VERDICT</div>
                <div class='{badge_class}' style='font-size:1.15rem; padding:6px 14px;'>{risk['verdict'].upper()}</div>
                <div style='font-size:0.8rem; color:#8b949e; margin-top:8px;'>Level: {risk['risk_level']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_score:
            st.markdown(f"""
            <div class='soc-card'>
                <div class='soc-card-header'>CYBER THREAT SCORE</div>
                <div class='metric-val' style='color:{risk['badge_color']};'>{risk['score']}<span style='font-size:1.1rem; color:#8b949e;'> / 100</span></div>
                <div class='risk-bar-container'>
                    <div class='risk-bar-fill' style='width:{risk['score']}%; background-color:{risk['badge_color']};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_ml:
            st.markdown(f"""
            <div class='soc-card'>
                <div class='soc-card-header'>NLP MODEL PROBABILITY</div>
                <div class='metric-val' style='color:#58a6ff;'>{cur['ml_probability']*100:.1f}%</div>
                <div style='font-size:0.8rem; color:#8b949e; margin-top:8px;'>Calibrated Platt Scaling</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_auth:
            auth_val = cur["auth_analysis"]["spf_status"]
            auth_col = "#f85149" if auth_val in ["FAIL", "SOFTFAIL"] else ("#3fb950" if auth_val == "PASS" else "#d29922")
            st.markdown(f"""
            <div class='soc-card'>
                <div class='soc-card-header'>SPF / DMARC RESULT</div>
                <div class='metric-val' style='color:{auth_col}; font-size:1.8rem;'>{auth_val}</div>
                <div style='font-size:0.8rem; color:#8b949e; margin-top:8px;'>DKIM: {cur['auth_analysis']['dkim_status']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📑 Itemized Contributing Risk Factors")
        st.markdown("Transparent mathematical score composition based on quantitative threat heuristics and statistical model weights:")
        
        factors_data = []
        for f in risk["factors"]:
            factors_data.append({
                "Risk Category": f["category"],
                "Points Added": f"+{f['points']} pts",
                "Max Range": f"{f['max']} pts",
                "Technical Finding / SOC Evidence": f["description"]
            })
        st.dataframe(pd.DataFrame(factors_data), use_container_width=True, hide_index=True)
        
        col_snippets, col_xai = st.columns([1, 1])
        
        with col_snippets:
            st.markdown("#### 🚨 Detected Suspicious Threat Snippets")
            if cur["snippets"]:
                for s in cur["snippets"]:
                    st.markdown(f"""
                    <div style='background-color:#161b22; border-left:3px solid #f85149; padding:10px 14px; margin-bottom:10px; border-radius:0 8px 8px 0; border:1px solid #30363d;'>
                        <span style='color:#ff7b72; font-weight:700; font-size:0.8rem;'>[{s['category'].upper()}]</span>
                        <code style='color:#58a6ff; font-weight:600;'>{s['matched_term']}</code>
                        <div style='color:#c9d1d9; font-size:0.85rem; margin-top:4px;'>{s['context']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No high-risk urgency or credential-harvesting keyword patterns located in body text.")
                
        with col_xai:
            st.markdown("#### 🧠 Top Statistical Predictive Tokens")
            st.markdown("Tokens in this specific email exerting the strongest statistical log-odds contribution toward Phishing:")
            triggers = cur["xai"]["top_phishing_triggers"]
            if triggers:
                trig_df = pd.DataFrame([
                    {
                        "Token / Phrase": t["token"],
                        "TF-IDF": t["tfidf"],
                        "Model Coef": t["weight"],
                        "Contribution": f"+{t['contribution']:.4f}"
                    }
                    for t in triggers
                ])
                st.dataframe(trig_df, use_container_width=True, hide_index=True)
            else:
                st.write("No strong phishing tokens activated.")
                
            legit_triggers = cur["xai"]["top_legitimate_indicators"]
            if legit_triggers:
                st.markdown("<div style='font-size:0.85rem; color:#3fb950; font-weight:600; margin-top:10px;'>Tokens pushing toward Legitimate:</div>", unsafe_allow_html=True)
                for lt in legit_triggers[:4]:
                    st.markdown(f"- `{lt['token']}` (Contribution: `{lt['contribution']:.4f}`)")


# ==============================================================================
# PAGE 3: ML DASHBOARD
# ==============================================================================
elif page == "3. ML Dashboard":
    st.markdown("### 🔬 Machine Learning Model Benchmark Dashboard")
    st.markdown("Comprehensive performance metrics, leakage-free cross-validation results, and algorithmic explainability.")
    
    comp = pipeline["comparison"]
    meta = pipeline["metadata"]
    
    # Overview KPIs
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='soc-card'><div class='soc-card-header'>WINNING CLASSIFIER</div><div style='font-size:1.3rem; font-weight:800; color:#58a6ff;'>{meta['best_model_name']}</div><div style='font-size:0.75rem; color:#8b949e; margin-top:6px;'>Platt Sigmoid Calibration</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='soc-card'><div class='soc-card-header'>TRAINING SAMPLES</div><div class='metric-val' style='color:#f0f6fc;'>{meta['train_samples']:,}</div><div style='font-size:0.75rem; color:#8b949e; margin-top:6px;'>80% Stratified Split</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='soc-card'><div class='soc-card-header'>HOLDOUT TEST SAMPLES</div><div class='metric-val' style='color:#f0f6fc;'>{meta['test_samples']:,}</div><div style='font-size:0.75rem; color:#8b949e; margin-top:6px;'>20% Unseen Evaluation</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='soc-card'><div class='soc-card-header'>TF-IDF VOCABULARY</div><div class='metric-val' style='color:#f0f6fc;'>{meta['tfidf_vocabulary_size']:,}</div><div style='font-size:0.75rem; color:#8b949e; margin-top:6px;'>Unigrams & Bigrams</div></div>", unsafe_allow_html=True)
        
    st.markdown("#### 🏆 4-Model Comparative Benchmark Table")
    st.markdown("All 4 models evaluated on identical holdout test set (3,727 emails) with 5-Fold Stratified Cross-Validation on the training set:")
    
    table_rows = []
    for model_name, metrics in comp.items():
        is_best = " ⭐ (Best)" if model_name == meta["best_model_name"] else ""
        table_rows.append({
            "Algorithm": model_name + is_best,
            "Accuracy": f"{metrics['Accuracy']*100:.2f}%",
            "Precision": f"{metrics['Precision']*100:.2f}%",
            "Recall": f"{metrics['Recall']*100:.2f}%",
            "F1-Score": f"{metrics['F1-Score']*100:.2f}%",
            "ROC-AUC": f"{metrics['ROC-AUC']:.4f}",
            "5-Fold CV F1": f"{metrics['CV_F1_Mean']*100:.2f}% (±{metrics['CV_F1_Std']*100:.2f}%)",
            "Training Time": f"{metrics['Training_Time_Sec']:.2f}s"
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    col_plot1, col_plot2 = st.columns([1, 1])
    with col_plot1:
        st.markdown("#### 🎯 Confusion Matrix (Winning Model)")
        cm_path = os.path.join(BASE_DIR, "models", "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
            best_m = comp[meta["best_model_name"]]
            st.caption(f"True Negatives: {best_m['True_Negative']:,} | False Positives: {best_m['False_Positive']} | False Negatives: {best_m['False_Negative']} | True Positives: {best_m['True_Positive']:,}")
            
    with col_plot2:
        st.markdown("#### 📈 Multi-Model ROC Comparison Curve")
        roc_path = os.path.join(BASE_DIR, "models", "roc_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
            st.caption("Demonstrates discriminative power across false positive vs true positive rates.")
            
    st.markdown("---")
    
    st.markdown("#### 🔑 Global Feature Importance (Top Indicative Phishing Terms)")
    feat_img_path = os.path.join(BASE_DIR, "models", "top_features.png")
    if os.path.exists(feat_img_path):
        st.image(feat_img_path, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 🧪 On-Demand LIME Local Surrogate Explanation")
    st.markdown("Run a live LIME perturbation experiment on any custom email text:")
    
    lime_input = st.text_area(
        "Enter text to perturb with LIME:",
        value="URGENT: Verify your account immediately at http://192.168.1.1/login or your banking access will be terminated within 24 hours.",
        height=90
    )
    if st.button("RUN LIME PERTURBATION EXPERIMENT"):
        with st.spinner("Generating 250 perturbed text samples and fitting local surrogate model..."):
            lime_res = explainer.explain_with_lime(lime_input, num_features=8, num_samples=250)
            
            c_l1, c_l2 = st.columns(2)
            with c_l1:
                st.markdown("**Terms Increasing Phishing Probability:**")
                for item in lime_res["phishing_cues"]:
                    st.markdown(f"- 🔴 `{item['word']}`: `+{item['weight']:.4f}`")
            with c_l2:
                st.markdown("**Terms Decreasing Phishing Probability:**")
                for item in lime_res["legitimate_cues"]:
                    st.markdown(f"- 🟢 `{item['word']}`: `{item['weight']:.4f}`")


# ==============================================================================
# PAGE 4: INVESTIGATION REPORT
# ==============================================================================
elif page == "4. Investigation Report":
    st.markdown("### 📋 Forensic Investigation Report")
    
    if st.session_state["current_analysis"] is None:
        st.warning("⚠️ No email analyzed yet. Please analyze an email in **1. Email Analyzer** to generate forensic findings.")
    else:
        cur = st.session_state["current_analysis"]
        parsed = cur["parsed_email"]
        iocs = cur["iocs"]
        sender = cur["sender_analysis"]
        auth = cur["auth_analysis"]
        risk = cur["risk_assessment"]
        
        st.markdown(f"""
        <div class='soc-card'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <span style='font-size:1.3rem; font-weight:800; color:#f0f6fc;'>CASE FILE: {cur['case_id']}</span>
                    <span style='margin-left:12px; font-size:0.75rem; background-color:#30363d; color:#58a6ff; padding:3px 10px; border-radius:4px; font-weight:700;'>TLP:AMBER</span>
                </div>
                <div style='font-size:0.85rem; color:#8b949e;'>Logged: {cur['timestamp']} UTC</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        pdf_bytes = generate_forensic_pdf(
            case_id=cur["case_id"],
            parsed_email=parsed,
            risk_assessment=risk,
            ml_analysis={"phishing_probability": cur["ml_probability"]},
            iocs=iocs,
            sender_analysis=sender,
            auth_analysis=auth,
            xai_explanation=cur["xai"]
        )
        
        st.download_button(
            label="📄 DOWNLOAD OFFICIAL PDF FORENSIC REPORT",
            data=pdf_bytes,
            file_name=f"{cur['case_id']}_threat_forensics.pdf",
            mime="application/pdf",
            type="primary"
        )
        
        st.markdown("---")
        st.markdown("#### 1. Extracted Email Headers & Authentication Security")
        c_h1, c_h2 = st.columns(2)
        with c_h1:
            st.markdown(f"**From:** `{parsed.get('from', 'N/A')}`")
            st.markdown(f"**Return-Path:** `{parsed.get('return_path', 'N/A')}`")
            st.markdown(f"**Reply-To:** `{parsed.get('reply_to', 'N/A')}`")
            st.markdown(f"**Subject:** `{parsed.get('subject', 'N/A')}`")
        with c_h2:
            st.markdown(f"**SPF Record:** `{auth['spf_status']}`")
            st.markdown(f"**DKIM Signature:** `{auth['dkim_status']}`")
            st.markdown(f"**DMARC Policy:** `{auth['dmarc_status']}`")
            st.markdown(f"**Message-ID:** `{parsed.get('message_id', 'N/A')}`")
            
        if sender["mismatches"]:
            st.markdown("##### 🚨 Sender Identity Spoofing Evidence:")
            for m in sender["mismatches"]:
                st.error(f"**[{m['type']}] ({m['severity']})**: {m['description']}")
                
        st.markdown("---")
        st.markdown("#### 2. Extracted Indicators of Compromise (Defanged)")
        col_urls, col_ips = st.columns(2)
        with col_urls:
            st.markdown(f"**Extracted URLs ({len(iocs['defanged_urls'])}):**")
            if iocs["defanged_urls"]:
                for u in iocs["defanged_urls"]:
                    st.markdown(f"- <code class='defanged-code'>{u}</code>", unsafe_allow_html=True)
            else:
                st.write("No external web URLs detected.")
                
        with col_ips:
            st.markdown(f"**Extracted IP Addresses ({len(iocs['defanged_ips'])}):**")
            if iocs["defanged_ips"]:
                for ip in iocs["defanged_ips"]:
                    st.markdown(f"- <code class='defanged-code'>{ip}</code>", unsafe_allow_html=True)
            else:
                st.write("No IP addresses detected.")
                
        st.markdown("---")
        st.markdown("#### 3. Attachment Safety Inspection")
        attachments = parsed.get("attachments", [])
        if attachments:
            att_data = []
            for a in attachments:
                att_data.append({
                    "Filename": a["filename"],
                    "MIME Type": a["content_type"],
                    "Size": f"{a['size_bytes']:,} bytes",
                    "Security Flag": "🚨 HIGH RISK EXEC" if a["is_risky"] else "✅ CLEAN"
                })
            st.dataframe(pd.DataFrame(att_data), use_container_width=True, hide_index=True)
        else:
            st.info("No attachments embedded in this message.")

    st.markdown("---")
    st.markdown("#### 📜 Session Investigation History Log")
    if st.session_state["history"]:
        history_table = []
        for h in st.session_state["history"]:
            history_table.append({
                "Case ID": h["case_id"],
                "Timestamp": h["timestamp"],
                "Subject": h["parsed_email"].get("subject", "N/A")[:35],
                "Verdict": h["risk_assessment"]["verdict"],
                "Risk Score": f"{h['risk_assessment']['score']}/100",
                "ML Probability": f"{h['ml_probability']*100:.1f}%",
                "Source": h["source_type"]
            })
        st.dataframe(pd.DataFrame(history_table), use_container_width=True, hide_index=True)
    else:
        st.write("No investigations logged in this session yet.")
