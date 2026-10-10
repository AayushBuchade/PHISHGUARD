# 🛡️ PHISHGUARD — AI-Powered Email Threat Detection

> **An Explainable Machine Learning & SOC Cyber Forensics System for Real-Time Email Threat Triage**  
> *Built with Python, Scikit-Learn, Sublinear TF-IDF, Calibrated Linear SVM, LIME XAI, and Streamlit.*

---

## 📌 Executive Summary
**PHISHGUARD** is an enterprise-grade, placement-portfolio machine learning application engineered to detect, classify, and explain email-based cyber threats (Phishing, Brand Impersonation, BEC Wire Fraud, and Credential Harvesting). 

Rather than relying on black-box heuristics or external paid APIs, PhishGuard trains and compares **four genuine machine learning algorithms** on a benchmark dataset of **18,650 verified real-world emails**. The champion classifier—a **Calibrated Linear Support Vector Machine (LinearSVC with Platt Scaling)**—achieves **97.16% Accuracy**, **96.42% F1-score**, and **0.9957 ROC-AUC**, delivering sub-2ms inference times and mathematically transparent token-level explainability.

---

## 🏗️ Architecture & ML Pipeline

```
+----------------------------------------------------------------------------------------------------+
|                                    PHISHGUARD THREAT PIPELINE                                      |
+----------------------------------------------------------------------------------------------------+
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
      [ Raw Pasted Email Text ]                                          [ Uploaded .EML File ]
                 │                                                                 │
                 └────────────────────────────────┬────────────────────────────────┘
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │      RFC 822 / 2822 Email Header Parser         │
                         │  - From, Return-Path, Reply-To Discrepancies    │
                         │  - SPF / DKIM / DMARC Header Records            │
                         │  - Attachment Metadata & Extension Inspection   │
                         └────────────────────────┬────────────────────────┘
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │   Cybersecurity Indicators of Compromise (IOC)  │
                         │  - Passive URL Extraction & Safe Defanging      │
                         │  - Direct IPv4 Address Host Detection           │
                         │  - High-Risk TLD Abuse (.xyz, .pw, .top, etc.)  │
                         └────────────────────────┬────────────────────────┘
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │           Domain Feature Engineering            │
                         │  - Urgency & Psychological Coercion Cues        │
                         │  - Credential Harvesting Triggers               │
                         │  - Financial Transaction & Wire Fraud Terms     │
                         │  - Brand Impersonation (PayPal, M365, Banks)    │
                         └────────────────────────┬────────────────────────┘
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │      Security-Aware Text Preprocessor & NLP     │
                         │  - Masking: <token_url>, <token_ip>, <currency> │
                         │  - Stopword filtering (protecting threat cues)  │
                         │  - Sublinear TF-IDF Vectorizer (1-2 N-grams)    │
                         └────────────────────────┬────────────────────────┘
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │   Calibrated ML Classifiers (Leakage-Free 80/20)│
                         │  1. Calibrated Linear SVM (Platt Sigmoid)       │
                         │  2. Logistic Regression (L2 Regularized)        │
                         │  3. Multinomial Naive Bayes                     │
                         │  4. Random Forest (100 Trees Ensemble)          │
                         └────────────────────────┬────────────────────────┘
                                                  ▼
                         ┌─────────────────────────────────────────────────┐
                         │       Composite Threat Risk Scoring (0 - 100)   │
                         │   Score = f(ML_Prob, Spoofing, IOCs, Payloads)  │
                         │   [0-34: Safe] | [35-69: Suspicious] | [70+: Phish]
                         └────────────────────────┬────────────────────────┘
                                                  ▼
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
     [ Explainable AI (LIME & TF-IDF) ]                              [ Downloadable PDF Report ]
     - Instance Log-Odds Attribution                                 - SOC Triage Forensic Audit
     - Highlighted Threat Snippets                                   - Official Remediation Playbook
```

---

## 📊 Real Model Benchmark Results
All models were trained on **14,907 emails** and evaluated on a holdout test partition of **3,727 emails** (80/20 stratified split) with **5-Fold Stratified Cross-Validation**:

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV F1 (Mean ± Std) | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** ⭐ | **97.16%** | **95.14%** | **97.74%** | **96.42%** | **0.9957** | **96.32% (±0.33%)** | 2.16s |
| **Logistic Regression** | 96.65% | 94.84% | 96.72% | 95.77% | 0.9950 | 95.79% (±0.33%) | 2.31s |
| **Random Forest** | 95.41% | 93.88% | 94.46% | 94.17% | 0.9902 | 93.85% (±0.28%) | 4.79s |
| **Multinomial Naive Bayes** | 94.98% | 93.46% | 93.78% | 93.62% | 0.9889 | 94.13% (±0.37%) | 1.49s |

### Holdout Confusion Matrix (Winning Model — Linear SVM):
- **True Negatives (TN):** `2,192` (Legitimate correctly cleared)
- **False Positives (FP):** `73` (Legitimate flagged as phish — 3.2% FPR)
- **False Negatives (FN):** `33` (Phishing missed — **only 2.2% FNR!**)
- **True Positives (TP):** `1,429` (Phishing correctly caught)

---

## 🗃️ Dataset Provenance & Citation
- **Dataset Size:** 18,650 labeled emails (11,322 Safe Emails, 7,328 Phishing Emails)
- **Sources:**
  1. **Nazario Phishing Corpus** (Dr. Jose Nazario): Authentic targeted spear phishing and credential harvesting emails collected for defensive research.
  2. **Enron Clean Email Corpus** (FERC / CMU): Real non-synthetic business communications.
  3. **SpamAssassin Public Corpus**: Public domain ham/spam samples.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) & Public Domain Dedication (PDDL).
- **Documentation:** See [`data/raw/DATASET_SOURCE.md`](file:///c:/Users/aayus/OneDrive/Desktop/Human%20firewall%20beta/data/raw/DATASET_SOURCE.md).

---

## 🛡️ Cybersecurity Analysis & Threat Scoring

PhishGuard outputs two distinct, complementary scores:
1. **ML Phishing Probability (0.0% to 100.0%):** Pure statistical linguistic probability derived from sublinear TF-IDF n-gram distributions.
2. **Cyber Threat Risk Score (0 to 100):** A transparent, deterministic composite metric computed by the SOC Security Engine:
   $$\text{Risk Score} = \min(100, \, \text{ML\_Points}_{[0-40]} + \text{Spoofing}_{[0-20]} + \text{IOC\_Threats}_{[0-20]} + \text{Social\_Eng}_{[0-10]} + \text{Payloads}_{[0-10]})$$

### Defensive Safety Guarantee
PhishGuard implements **strict passive static analysis**:
- Extracted URLs and IP addresses are defanged before UI rendering (`hxxp://`, `192[.]168[.]1[.]1`).
- No active HTTP/DNS requests are made to untrusted domains.
- File attachments are parsed for MIME types and dangerous extensions (`.exe`, `.scr`, `.docm`) without execution.

---

## 🔍 Explainable AI (XAI)
1. **Direct Mathematical Token Attribution:**
   $$\text{Contribution}_i = \text{TF-IDF}(x_i) \times w_i$$
   Instantly reveals which words in the specific email exerted positive (phishing) or negative (safe) log-odds pressure.
2. **LIME (Local Interpretable Model-agnostic Explanations):**
   Generates localized surrogate models by randomly perturbing email tokens to prove algorithmic sensitivity.

---

## 🖥️ Streamlit Application Interface

The UI is inspired by modern Security Operations Center (SOC) dark consoles (`#0b0f19` charcoal theme):
1. **Email Analyzer (`Page 1`):** Paste raw text or upload `.eml` files; 1-click sample loaders (PayPal phishing, CEO wire fraud, sprint notes).
2. **Risk Assessment (`Page 2`):** Executive KPI cards, itemized contributing factors breakdown table, detected suspicious snippets.
3. **ML Dashboard (`Page 3`):** 4-model comparison table, confusion matrix heatmap, multi-model ROC curve, global feature importance chart, interactive on-demand LIME perturbation tool.
4. **Investigation Report (`Page 4`):** RFC header inspector, defanged IOC ledger, session investigation history log, and **1-click downloadable official PDF forensic report**.

---

## 📁 Repository Directory Structure

```
Human firewall beta/
├── app.py                         # Streamlit Cyber SOC Dark UI
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Git ignore rules
├── README.md                      # Complete portfolio project documentation
├── INTERVIEW_GUIDE.md             # In-depth technical interview guide
│
├── data/
│   └── raw/
│       ├── Phishing_Email.csv     # 18,650 benchmark email corpus
│       └── DATASET_SOURCE.md      # Provenance, license, and citations
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Data ingestion and verification
│   ├── text_preprocessor.py       # Security-aware text cleaning & token masking
│   ├── feature_engineering.py     # 13 domain cybersecurity heuristics
│   ├── security_analyzer.py       # EML parsing, header inspection, IOCs, risk scoring
│   ├── model_trainer.py           # Leakage-free training, 5-fold CV, calibration
│   ├── explainability.py          # LIME & TF-IDF linear weight attribution
│   └── report_generator.py        # Publication-quality PDF forensic report generator
│
├── models/
│   ├── best_model.joblib          # Selected champion model (Calibrated Linear SVM)
│   ├── tfidf_vectorizer.joblib    # Fitted TF-IDF vectorizer (5,000 n-grams)
│   ├── all_models.joblib          # All 4 trained classifiers
│   ├── model_comparison.json      # Benchmark metrics across all models
│   ├── model_metadata.json        # Training hyperparameters & metadata
│   ├── feature_importance.json    # Top 30 phishing vs safe tokens
│   ├── confusion_matrix.png       # Seaborn confusion matrix visualization
│   ├── roc_curve.png              # Multi-model ROC comparison plot
│   └── top_features.png           # Feature coefficient bar chart
│
├── notebooks/
│   └── phishguard_eda_and_training.ipynb # Full EDA, CV, and training Jupyter notebook
│
├── samples/
│   ├── phishing_account_suspended.eml    # Sample PayPal account lock phish
│   ├── phishing_urgent_wire_transfer.eml # Sample CEO fraud BEC wire lure
│   └── legitimate_project_update.eml     # Sample legitimate corporate update
│
├── tests/
│   └── test_e2e.py                # End-to-end unit and integration test suite
│
└── scripts/
    └── generate_notebook.py       # Notebook generator script
```

---

## 🚀 Quickstart & Setup (Windows / Local)

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows PowerShell or Command Prompt

### 2. Clone / Navigate to Directory
```powershell
cd "c:\Users\aayus\OneDrive\Desktop\Human firewall beta"
```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Run Automated End-to-End Tests
```powershell
python -m tests.test_e2e
```
*Expected Output:* `ALL 6 END-TO-END VERIFICATION TESTS PASSED SUCCESSFULLY!`

### 5. Launch the Streamlit Application
```powershell
python -m streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 📜 Placement & Portfolio Interview Ready
Check [`INTERVIEW_GUIDE.md`](file:///c:/Users/aayus/OneDrive/Desktop/Human%20firewall%20beta/INTERVIEW_GUIDE.md) for detailed architectural justifications, algorithmic trade-offs (TF-IDF vs LLMs, SVM vs Neural Nets), probability calibration mathematics, and cybersecurity incident response workflows.
