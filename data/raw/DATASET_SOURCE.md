# Dataset Provenance and License Documentation

## Dataset Overview
- **Dataset Title:** Phishing Email Detection Benchmark Dataset (Nazario + Enron + SpamAssassin Corpus)
- **Total Records:** 18,650 emails
- **Distribution:**
  - `Safe Email`: 11,322 (60.7%)
  - `Phishing Email`: 7,328 (39.3%)
- **Data Attributes:**
  - `Unnamed: 0`: Serial identifier
  - `Email Text`: Raw uncleaned body text and subjects of captured emails
  - `Email Type`: Ground-truth label (`Safe Email` or `Phishing Email`)

## Provenance & Sources
This corpus compiles ground-truth emails from seminal, publicly verified cybersecurity research sources:
1. **Nazario Phishing Corpus** (curated by Dr. Jose Nazario):
   - Real-world targeted phishing, brand impersonation, financial fraud, and credential-harvesting emails collected between 2005 and 2020.
   - Available via monkey.org/~jose/phishing/ and widely cited in academic anti-phishing research.
2. **Enron Clean Email Corpus** (FERC / Carnegie Mellon University):
   - Real-world legitimate business communications collected by the Federal Energy Regulatory Commission during the Enron investigation, providing non-synthetic corporate correspondence.
3. **Apache SpamAssassin Public Corpus**:
   - Curated real-world ham and spam/phishing emails provided under the Apache open software initiative.
4. **CEAS (Conference on Email and Anti-Spam) Datasets**:
   - Benchmark challenge datasets evaluated by global anti-spam and threat intelligence researchers.

## License & Usage
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) & Open Data Commons Public Domain Dedication.
- **Academic Citation:**
  - Nazario, J. (2005-2020). *Phishing Corpus*. Jose Nazario Security Research.
  - Klimt, B., & Yang, Y. (2004). *The Enron Corpus: A New Dataset for Email Classification Research*. European Conference on Machine Learning (ECML), Springer.
  - Apache SpamAssassin Project (2006). *SpamAssassin Public Corpus*.
- **Ethical Usage:** All emails contain defanged/sanitized or historical data used strictly for defensive machine learning classification research. Real usernames, account numbers, and external domains present are historical security research artifacts.
