"""
PhishGuard - Machine Learning Training & Evaluation Pipeline
Trains and compares 4 algorithms on real phishing email data:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest Classifier
4. Calibrated Linear Support Vector Machine (LinearSVC with Platt Scaling)

Enforces:
- Leakage-free train/test split (80/20 stratified)
- 5-Fold Stratified Cross-Validation
- Calibrated probability estimates across all models
- Full suite of metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Generation of confusion matrices, ROC curves, and feature importance charts
- Persistence via Joblib and JSON metadata
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless backend for server/CLI rendering
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from src.text_preprocessor import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Phishing_Email.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_prepare_dataset(csv_path: str = DATA_PATH):
    """
    Loads raw dataset, removes nulls, maps labels to binary targets.
    0 = Safe Email (Legitimate)
    1 = Phishing Email (Malicious)
    """
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Raw shape: {df.shape}")
    
    # Drop rows missing text
    df = df.dropna(subset=['Email Text'])
    df['Email Text'] = df['Email Text'].astype(str)
    
    # Encode labels
    label_map = {"Safe Email": 0, "Phishing Email": 1}
    df['target'] = df['Email Type'].map(label_map)
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    
    print("Cleaned shape:", df.shape)
    print("Class counts:\n", df['target'].value_counts())
    return df


def train_and_evaluate_all():
    """
    Executes the complete leakage-free training, CV, evaluation, and serialization pipeline.
    """
    start_total_time = time.time()
    df = load_and_prepare_dataset()
    
    # Clean text using our security-aware NLP preprocessor
    print("Cleaning text corpus...")
    t0 = time.time()
    # Apply clean_text
    X_raw = df['Email Text'].apply(clean_text)
    y = df['target'].values
    print(f"Corpus cleaned in {time.time() - t0:.2f}s")
    
    # 1. Leakage-Free Train/Test Split (80% Train, 20% Test, Stratified)
    print("Performing stratified train/test split (80/20)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train_raw):,} samples | Test size: {len(X_test_raw):,} samples")
    
    # 2. Fit TF-IDF strictly on Train, Transform Test (Leakage-free)
    print("Fitting TF-IDF Vectorizer on training partition...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=3,
        max_df=0.9
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    
    # 3. Model Definitions
    # Note: LinearSVC is calibrated with 3-fold CV using Platt scaling (sigmoid)
    raw_models = {
        "Logistic Regression": LogisticRegression(C=1.5, max_iter=1000, random_state=42, solver='lbfgs'),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42, n_jobs=-1),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, max_iter=2000, random_state=42),
            method='sigmoid',
            cv=3
        )
    }
    
    results = {}
    trained_models = {}
    roc_data = {}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "="*70)
    print("TRAINING & COMPARING MODELS WITH 5-FOLD CROSS-VALIDATION")
    print("="*70)
    
    for name, clf in raw_models.items():
        print(f"\n---> Training: {name}...")
        t_start = time.time()
        
        # 5-fold cross-validation on training data
        cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=skf, scoring='f1', n_jobs=-1)
        mean_cv_f1 = float(np.mean(cv_scores))
        std_cv_f1 = float(np.std(cv_scores))
        
        # Fit on full training set
        clf.fit(X_train_tfidf, y_train)
        fit_time = time.time() - t_start
        trained_models[name] = clf
        
        # Predictions & Probabilities on Holdout Test Set
        y_pred = clf.predict(X_test_tfidf)
        
        # Predict Proba (all models including calibrated SVM support predict_proba)
        y_prob = clf.predict_proba(X_test_tfidf)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist() # [[TN, FP], [FN, TP]]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}
        
        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(roc_auc), 4),
            "CV_F1_Mean": round(mean_cv_f1, 4),
            "CV_F1_Std": round(std_cv_f1, 4),
            "Training_Time_Sec": round(fit_time, 2),
            "Confusion_Matrix": cm,
            "True_Negative": cm[0][0],
            "False_Positive": cm[0][1],
            "False_Negative": cm[1][0],
            "True_Positive": cm[1][1]
        }
        
        print(f"     Accuracy:  {acc*100:.2f}% | Precision: {prec*100:.2f}%")
        print(f"     Recall:    {rec*100:.2f}% | F1-Score:  {f1*100:.2f}%")
        print(f"     ROC-AUC:   {roc_auc:.4f}  | 5-Fold CV F1: {mean_cv_f1*100:.2f}% (+/- {std_cv_f1*100:.2f}%)")
        print(f"     Fit Time:  {fit_time:.2f}s")
        print(f"     Confusion: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # 4. Select the Best Model (Highest F1-score & ROC-AUC)
    best_name = max(results, key=lambda k: (results[k]["F1-Score"], results[k]["ROC-AUC"]))
    best_model = trained_models[best_name]
    print("\n" + "="*70)
    print(f"WINNING MODEL SELECTED: {best_name}")
    print(f"F1-Score: {results[best_name]['F1-Score']:.4f} | ROC-AUC: {results[best_name]['ROC-AUC']:.4f}")
    print("="*70)
    
    # 5. Extract Top Feature Importances / Coefficients
    print("\nExtracting feature importance & top indicative terms...")
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # If best is Linear SVM or Logistic Regression, get exact coefficients
    lr_model = trained_models.get("Logistic Regression")
    lr_coefs = lr_model.coef_[0]
    
    top_phish_idx = np.argsort(lr_coefs)[-30:][::-1]
    top_legit_idx = np.argsort(lr_coefs)[:30]
    
    feature_importance = {
        "top_phishing_tokens": [
            {"token": str(feature_names[i]), "weight": round(float(lr_coefs[i]), 4)}
            for i in top_phish_idx
        ],
        "top_legitimate_tokens": [
            {"token": str(feature_names[i]), "weight": round(float(lr_coefs[i]), 4)}
            for i in top_legit_idx
        ]
    }
    
    # 6. Save Plots: Confusion Matrix & ROC Curves
    print("Generating evaluation plots...")
    # Plot 1: Best Model Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm_best = np.array(results[best_name]["Confusion_Matrix"])
    sns.heatmap(
        cm_best, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Safe Email', 'Phishing'],
        yticklabels=['Safe Email', 'Phishing'],
        cbar=False
    )
    plt.title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("Actual Class", fontsize=10)
    plt.tight_layout()
    cm_plot_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    
    # Plot 2: Model Comparison ROC Curves
    plt.figure(figsize=(7, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (m_name, r_info) in enumerate(roc_data.items()):
        plt.plot(
            r_info["fpr"], r_info["tpr"],
            label=f"{m_name} (AUC = {r_info['auc']:.4f})",
            linewidth=2, color=colors[idx % len(colors)]
        )
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label="Random Guess (AUC = 0.50)")
    plt.xlabel("False Positive Rate (FPR)", fontsize=10)
    plt.ylabel("True Positive Rate (Recall)", fontsize=10)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_plot_path = os.path.join(MODELS_DIR, "roc_curve.png")
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()
    
    # Plot 3: Top Feature Coefficients Barplot
    plt.figure(figsize=(10, 6))
    top_terms = [item["token"] for item in feature_importance["top_phishing_tokens"][:15]]
    top_weights = [item["weight"] for item in feature_importance["top_phishing_tokens"][:15]]
    sns.barplot(x=top_weights, y=top_terms, palette="Reds_r")
    plt.title("Top Phishing Predictive Feature Weights (TF-IDF Log-Odds)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Coefficient Weight", fontsize=10)
    plt.ylabel("Term / Bigram", fontsize=10)
    plt.tight_layout()
    feat_plot_path = os.path.join(MODELS_DIR, "top_features.png")
    plt.savefig(feat_plot_path, dpi=200)
    plt.close()
    
    # 7. Persist Artifacts via Joblib & JSON
    print("Serializing trained models and metadata...")
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    all_models_path = os.path.join(MODELS_DIR, "all_models.joblib")
    comparison_json_path = os.path.join(MODELS_DIR, "model_comparison.json")
    metadata_json_path = os.path.join(MODELS_DIR, "model_metadata.json")
    feature_imp_path = os.path.join(MODELS_DIR, "feature_importance.json")
    
    joblib.dump(best_model, best_model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(trained_models, all_models_path)
    
    with open(comparison_json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    with open(feature_imp_path, "w") as f:
        json.dump(feature_importance, f, indent=4)
        
    metadata = {
        "best_model_name": best_name,
        "total_dataset_samples": int(len(df)),
        "safe_email_count": int(sum(df['target'] == 0)),
        "phishing_email_count": int(sum(df['target'] == 1)),
        "train_samples": int(len(X_train_raw)),
        "test_samples": int(len(X_test_raw)),
        "tfidf_vocabulary_size": int(len(feature_names)),
        "ngram_range": [1, 2],
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_pipeline_time_seconds": round(time.time() - start_total_time, 2),
        "best_model_metrics": results[best_name]
    }
    with open(metadata_json_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\nModel training & serialization complete!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    print(f"Metrics metadata saved to: {metadata_json_path}")
    return results, metadata

if __name__ == "__main__":
    train_and_evaluate_all()
