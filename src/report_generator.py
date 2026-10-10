"""
PhishGuard - Forensic PDF Report Generator
Generates publication-quality, SOC-grade downloadable PDF investigation reports
using FPDF. Sanitizes strings to Latin-1 compatibility to prevent encoding errors.
Includes Incident Metadata, Executive Verdict, Risk Score, Headers, Extracted IOCs,
XAI Contributing Factors, and Remediation Guidance.
"""

import time
from typing import Dict, Any, List
from fpdf import FPDF


def sanitize_text(text: Any) -> str:
    """Sanitizes unicode characters into Latin-1 compatible representations for FPDF."""
    if text is None:
        return ""
    s = str(text)
    # Replace common unicode typographic characters with ASCII equivalents
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2014": "-", "\u2013": "-", "\u2026": "...", "\u2022": "*",
        "\xa0": " ", "\u20ac": "EUR", "\u00a3": "GBP", "\u00a5": "YEN"
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # Filter strictly to latin-1
    return s.encode("latin-1", "replace").decode("latin-1")


class ForensicReportPDF(FPDF):
    """Custom styled PDF document for PhishGuard forensic reports."""
    
    def header(self):
        # Top banner with dark styling
        self.set_fill_color(15, 23, 42) # Dark navy
        self.rect(0, 0, 210, 24, 'F')
        
        self.set_xy(10, 4)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(248, 250, 252) # White
        self.cell(0, 8, sanitize_text("PHISHGUARD — EMAIL THREAT FORENSICS REPORT"), ln=1)
        
        self.set_xy(10, 13)
        self.set_font('Arial', '', 8)
        self.set_text_color(148, 163, 184) # Muted slate
        self.cell(0, 6, sanitize_text("CONFIDENTIAL // SOC TIER-2 INCIDENT TRIAGE // TLP:AMBER"), ln=1)
        
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, sanitize_text(f"PhishGuard AI Incident Response | Generated {time.strftime('%Y-%m-%d %H:%M:%S')} | Page {self.page_no()}"), align='C')


def generate_forensic_pdf(
    case_id: str,
    parsed_email: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    ml_analysis: Dict[str, Any],
    iocs: Dict[str, Any],
    sender_analysis: Dict[str, Any],
    auth_analysis: Dict[str, Any],
    xai_explanation: Dict[str, Any]
) -> bytes:
    """
    Assembles all forensic findings into a multi-section PDF document.
    Returns the PDF as raw bytes ready for Streamlit download button.
    """
    pdf = ForensicReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Section 1: Incident Meta & Executive Verdict
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, sanitize_text("1. EXECUTIVE INCIDENT SUMMARY"), ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    # Meta box
    pdf.set_fill_color(241, 245, 249)
    pdf.rect(10, pdf.get_y(), 190, 28, 'F')
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(30, 41, 59)
    
    y_start = pdf.get_y() + 2
    pdf.set_xy(14, y_start)
    pdf.cell(90, 5, sanitize_text(f"Case Reference: {case_id}"), ln=0)
    pdf.cell(90, 5, sanitize_text(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}"), ln=1)
    
    pdf.set_x(14)
    pdf.cell(90, 5, sanitize_text(f"Subject: {parsed_email.get('subject', 'N/A')[:45]}"), ln=0)
    pdf.cell(90, 5, sanitize_text(f"Sender: {parsed_email.get('from', 'N/A')[:45]}"), ln=1)
    
    pdf.set_x(14)
    pdf.cell(90, 5, sanitize_text(f"Engine: Linear SVM (Calibrated) & SOC Heuristics"), ln=0)
    pdf.cell(90, 5, sanitize_text(f"Risk Classification: {risk_assessment.get('risk_level', 'UNKNOWN')}"), ln=1)
    
    pdf.set_xy(10, y_start + 30)
    
    # Verdict KPI Grid
    verdict = risk_assessment.get("verdict", "Unknown")
    risk_score = risk_assessment.get("score", 0)
    ml_prob = ml_analysis.get("phishing_probability", 0.0) * 100
    
    pdf.set_font('Arial', 'B', 10)
    # Verdict Badge
    if "Phish" in verdict:
        pdf.set_fill_color(254, 226, 226) # light red
        pdf.set_text_color(185, 28, 28) # dark red
    elif "Suspicious" in verdict:
        pdf.set_fill_color(254, 243, 199) # light amber
        pdf.set_text_color(180, 83, 9) # dark amber
    else:
        pdf.set_fill_color(220, 252, 231) # light green
        pdf.set_text_color(21, 128, 61) # dark green
        
    pdf.cell(60, 10, sanitize_text(f"VERDICT: {verdict.upper()}"), border=1, fill=True, align='C', ln=0)
    
    pdf.set_fill_color(248, 250, 252)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(65, 10, sanitize_text(f"Risk Score: {risk_score} / 100"), border=1, fill=True, align='C', ln=0)
    pdf.cell(65, 10, sanitize_text(f"ML Probability: {ml_prob:.1f}%"), border=1, fill=True, align='C', ln=1)
    
    pdf.ln(6)
    
    # Section 2: Header Authenticity & Sender Spoofing
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, sanitize_text("2. HEADER AUTHENTICITY & SENDER VERIFICATION"), ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(30, 41, 59)
    
    headers_info = [
        ("From Header", parsed_email.get("from", "N/A")),
        ("Return-Path", parsed_email.get("return_path", "N/A") or "Not Specified"),
        ("Reply-To", parsed_email.get("reply_to", "N/A") or "Not Specified"),
        ("SPF Record", auth_analysis.get("spf_status", "NOT FOUND")),
        ("DKIM Signature", auth_analysis.get("dkim_status", "NOT FOUND")),
        ("DMARC Policy", auth_analysis.get("dmarc_status", "NOT FOUND"))
    ]
    for label, val in headers_info:
        pdf.set_font('Arial', 'B', 8)
        pdf.cell(35, 5, sanitize_text(f"{label}:"), ln=0)
        pdf.set_font('Arial', '', 8)
        pdf.cell(0, 5, sanitize_text(str(val)[:80]), ln=1)
        
    mismatches = sender_analysis.get("mismatches", [])
    if mismatches:
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 8)
        pdf.set_text_color(185, 28, 28)
        pdf.cell(0, 5, sanitize_text("[!] SENDER SPOOFING DETECTED:"), ln=1)
        pdf.set_font('Arial', '', 8)
        for m in mismatches:
            pdf.cell(0, 5, sanitize_text(f"  * {m['type']} [{m['severity']}]: {m['description']}"), ln=1)
    pdf.set_text_color(30, 41, 59)
    pdf.ln(6)
    
    # Section 3: Extracted Indicators of Compromise (IOCs)
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, sanitize_text("3. EXTRACTED INDICATORS OF COMPROMISE (DEFANGED)"), ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    defanged_urls = iocs.get("defanged_urls", [])
    defanged_ips = iocs.get("defanged_ips", [])
    attachments = parsed_email.get("attachments", [])
    
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 5, sanitize_text(f"Extracted URLs / Links ({len(defanged_urls)} total):"), ln=1)
    pdf.set_font('Arial', '', 8)
    if defanged_urls:
        for u in defanged_urls[:6]:
            pdf.cell(0, 4.5, sanitize_text(f"  * {u[:95]}"), ln=1)
        if len(defanged_urls) > 6:
            pdf.cell(0, 4.5, sanitize_text(f"  * ... ({len(defanged_urls) - 6} more links omitted for brevity)"), ln=1)
    else:
        pdf.cell(0, 4.5, sanitize_text("  * No external web links detected in message body."), ln=1)
        
    pdf.ln(2)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 5, sanitize_text(f"Extracted IP Addresses ({len(defanged_ips)} total):"), ln=1)
    pdf.set_font('Arial', '', 8)
    if defanged_ips:
        pdf.cell(0, 4.5, sanitize_text(f"  * {', '.join(defanged_ips[:8])}"), ln=1)
    else:
        pdf.cell(0, 4.5, sanitize_text("  * No explicit IPv4 addresses embedded in text."), ln=1)
        
    pdf.ln(2)
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 5, sanitize_text(f"Attachments ({len(attachments)} total):"), ln=1)
    pdf.set_font('Arial', '', 8)
    if attachments:
        for a in attachments:
            flag = "[HIGH RISK EXTENSION]" if a.get("is_risky") else "[CLEAN]"
            pdf.cell(0, 4.5, sanitize_text(f"  * {a['filename']} ({a['size_bytes']} bytes) - {flag}"), ln=1)
    else:
        pdf.cell(0, 4.5, sanitize_text("  * No binary attachments included with message."), ln=1)
        
    pdf.ln(6)
    
    # Section 4: Risk Factor Breakdown Table
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, sanitize_text("4. THREAT RISK FACTOR BREAKDOWN"), ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    factors = risk_assessment.get("factors", [])
    pdf.set_fill_color(226, 232, 240)
    pdf.set_font('Arial', 'B', 8)
    pdf.cell(60, 6, sanitize_text("Factor Category"), border=1, fill=True, ln=0)
    pdf.cell(20, 6, sanitize_text("Points"), border=1, fill=True, align='C', ln=0)
    pdf.cell(110, 6, sanitize_text("Technical Finding"), border=1, fill=True, ln=1)
    
    pdf.set_font('Arial', '', 8)
    for f in factors:
        pdf.cell(60, 5.5, sanitize_text(f["category"][:32]), border=1, ln=0)
        pdf.cell(20, 5.5, sanitize_text(f"+{f['points']} pts"), border=1, align='C', ln=0)
        pdf.cell(110, 5.5, sanitize_text(f["description"][:68]), border=1, ln=1)
        
    pdf.ln(6)
    
    # Section 5: Explainable AI & SOC Recommendations
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, sanitize_text("5. EXPLAINABLE AI ATTRIBUTION & SOC ACTION PLAN"), ln=1)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    top_triggers = xai_explanation.get("top_phishing_triggers", [])
    if top_triggers:
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(0, 5, sanitize_text("Top Machine Learning Phishing Predictive Tokens:"), ln=1)
        pdf.set_font('Arial', '', 8)
        trig_str = ", ".join([f"'{t['token']}' (weight: +{t['contribution']:.3f})" for t in top_triggers[:6]])
        pdf.cell(0, 4.5, sanitize_text(f"  * {trig_str}"), ln=1)
        pdf.ln(2)
        
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(0, 5, sanitize_text("Recommended Incident Response Playbook:"), ln=1)
    pdf.set_font('Arial', '', 8)
    if "Phish" in verdict:
        actions = [
            "1. Block sender domain and envelope return-path at Secure Email Gateway (SEG).",
            "2. Add extracted defanged URLs and domain IOCs to organizational EDR/DNS blocklists.",
            "3. Purge/claw back this message from all recipient inboxes across the tenant.",
            "4. If recipient interacted with links or attachments, initiate immediate password reset & host isolate."
        ]
    elif "Suspicious" in verdict:
        actions = [
            "1. Route message to administrative quarantine for manual Tier-2 SOC inspection.",
            "2. Request out-of-band verification from the claimed sender through verified directory telephone.",
            "3. Inspect sending IP reputation in threat intelligence feeds (VirusTotal / AbuseIPDB)."
        ]
    else:
        actions = [
            "1. Message classified as legitimate business communication. No defensive blocking required.",
            "2. Standard email gateway TLS encryption and perimeter policy apply."
        ]
    for act in actions:
        pdf.cell(0, 4.5, sanitize_text(f"  {act}"), ln=1)
        
    # Return as raw bytes
    return pdf.output(dest='S').encode('latin-1')


if __name__ == "__main__":
    test_pdf_bytes = generate_forensic_pdf(
        case_id="PG-20261010-001",
        parsed_email={"subject": "Account Suspended", "from": "Security <alert@fake-paypal.xyz>", "attachments": []},
        risk_assessment={"verdict": "Phishing Threat", "score": 88, "risk_level": "HIGH CRITICAL RISK", "factors": [
            {"category": "Machine Learning Prediction", "points": 38.0, "description": "NLP Phishing probability: 95.0%"},
            {"category": "Brand Impersonation", "points": 15.0, "description": "Display name claims PayPal"}
        ]},
        ml_analysis={"phishing_probability": 0.95},
        iocs={"defanged_urls": ["hxxp://192[.]168[.]1[.]1/login"], "defanged_ips": ["192[.]168[.]1[.]1"]},
        sender_analysis={"mismatches": [{"type": "Lookalike Domain", "severity": "HIGH", "description": "Fake PayPal"}]},
        auth_analysis={"spf_status": "FAIL", "dkim_status": "NOT FOUND", "dmarc_status": "NOT FOUND"},
        xai_explanation={"top_phishing_triggers": [{"token": "verify account", "contribution": 0.42}]}
    )
    print("Forensic PDF successfully generated! Bytes length:", len(test_pdf_bytes))
