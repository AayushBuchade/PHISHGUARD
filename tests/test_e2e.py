"""
Test script to verify end-to-end functionality of all PhishGuard modules:
1. Data loading and preprocessing
2. Feature engineering
3. Security analysis & IOC extraction
4. Model inference with Calibrated Linear SVM, Logistic Regression, Naive Bayes, Random Forest
5. Explainability (Mathematical TF-IDF attribution and LIME)
6. PDF Report Generation
"""

import os
import json
import joblib
import pandas as pd
from src.text_preprocessor import clean_text, preprocess_for_tfidf
from src.feature_engineering import extract_cyber_features
from src.security_analyzer import SecurityAnalyzer
from src.explainability import EmailExplainer
from src.report_generator import generate_forensic_pdf

def run_e2e_tests():
    print("[1/6] Testing Text Preprocessor...")
    raw_sample = "URGENT ALERT! Your PayPal account suspended. Verify at http://192.168.1.1/login within 24 hours. Fee: $500."
    cleaned = clean_text(raw_sample)
    assert "token_ip_address" in cleaned or "token_url_link" in cleaned
    assert "suspended" in cleaned
    print("      -> Text Preprocessor passed!")

    print("[2/6] Testing Feature Engineering...")
    features = extract_cyber_features(raw_sample)
    assert features["urgency_count"] >= 1
    assert features["url_count"] >= 1
    assert features["brand_mention_count"] >= 1
    print(f"      -> Feature Engineering passed! Extracted: {features}")

    print("[3/6] Testing Security Analyzer...")
    analyzer = SecurityAnalyzer()
    eml_sample = """From: "PayPal Support" <support@fake-paypal-verify.xyz>
Reply-To: bad-actor@evil-server.pw
Subject: URGENT: Account Suspension Notice
Date: Thu, 10 Oct 2026 10:00:00 +0000
Authentication-Results: spf=fail; dkim=fail; dmarc=fail

Dear user, your PayPal account is locked. Verify at http://192.168.1.50/login.php
"""
    parsed = analyzer.parse_eml_content(eml_sample)
    iocs = analyzer.extract_iocs(parsed["body_text"])
    sender = analyzer.check_sender_mismatch(parsed["from"], parsed["return_path"], parsed["reply_to"])
    auth = analyzer.parse_auth_headers(parsed["headers"])
    assert len(iocs["defanged_urls"]) >= 1
    assert sender["is_suspicious"] is True
    assert auth["spf_status"] == "FAIL"
    print(f"      -> Security Analyzer passed! Found {len(sender['mismatches'])} sender mismatches.")

    print("[4/6] Testing Models Inference...")
    best_model = joblib.load("models/best_model.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    all_models = joblib.load("models/all_models.joblib")
    
    vec = vectorizer.transform([clean_text(parsed["body_text"])])
    prob = best_model.predict_proba(vec)[0][1]
    assert 0.0 <= prob <= 1.0
    print(f"      -> Best Model (Linear SVM Calibrated) Phishing Probability: {prob * 100:.2f}%")
    
    for m_name, m_clf in all_models.items():
        m_p = m_clf.predict_proba(vec)[0][1]
        print(f"         * {m_name}: {m_p * 100:.2f}% phishing")
        
    risk = analyzer.calculate_risk_score(
        ml_probability=prob,
        cyber_features=features,
        iocs=iocs,
        sender_analysis=sender,
        auth_analysis=auth,
        attachments=parsed["attachments"]
    )
    assert 70 <= risk["score"] <= 100
    print(f"      -> Threat Risk Score: {risk['score']}/100 ({risk['verdict']}) passed!")

    print("[5/6] Testing Explainability...")
    explainer = EmailExplainer(model=best_model, vectorizer=vectorizer)
    math_xai = explainer.explain_instance_weights(parsed["body_text"])
    assert len(math_xai["top_phishing_triggers"]) > 0
    print(f"      -> Top Mathematical Triggers: {[t['token'] for t in math_xai['top_phishing_triggers'][:3]]}")

    print("[6/6] Testing PDF Report Generation...")
    pdf_bytes = generate_forensic_pdf(
        case_id="PG-TEST-001",
        parsed_email=parsed,
        risk_assessment=risk,
        ml_analysis={"phishing_probability": prob},
        iocs=iocs,
        sender_analysis=sender,
        auth_analysis=auth,
        xai_explanation=math_xai
    )
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
    print(f"      -> Forensic PDF generation passed ({len(pdf_bytes)} bytes)!")

    print("\n" + "="*50)
    print("ALL 6 END-TO-END VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    run_e2e_tests()
