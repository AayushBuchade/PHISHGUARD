"""
PhishGuard - Explainable AI (XAI) Module
Provides dual-mode interpretability for email threat classifications:
1. Direct TF-IDF Linear Model Attribution (instantaneous token-level contribution)
2. LIME (Local Interpretable Model-agnostic Explanations) TextExplainer
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Optional
from lime.lime_text import LimeTextExplainer
from src.text_preprocessor import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class EmailExplainer:
    """
    Handles explainability for PhishGuard models.
    Translates high-dimensional statistical representations into human-actionable SOC findings.
    """
    
    def __init__(self, model=None, vectorizer=None):
        if model is None:
            model_path = os.path.join(MODELS_DIR, "best_model.joblib")
            self.model = joblib.load(model_path)
        else:
            self.model = model
            
        if vectorizer is None:
            vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
            self.vectorizer = joblib.load(vec_path)
        else:
            self.vectorizer = vectorizer
            
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.lime_explainer = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])
        
        # Extract underlying linear weights if available
        self.weights = self._extract_model_weights()
        
    def _extract_model_weights(self) -> Optional[np.ndarray]:
        """Extracts weight vector from LogisticRegression, LinearSVC or CalibratedClassifierCV."""
        if hasattr(self.model, "coef_"):
            return self.model.coef_[0]
        elif hasattr(self.model, "calibrated_classifiers_"):
            # Average base estimators from CalibratedClassifierCV
            weights = []
            for cc in self.model.calibrated_classifiers_:
                base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
                if hasattr(base, "coef_"):
                    weights.append(base.coef_[0])
            if weights:
                return np.mean(weights, axis=0)
        return None

    def explain_instance_weights(self, raw_text: str, top_k: int = 12) -> Dict[str, Any]:
        """
        Instantaneous mathematical attribution:
        Computes token contribution = tfidf_value * feature_weight for all active tokens.
        """
        cleaned = clean_text(raw_text)
        tfidf_vec = self.vectorizer.transform([cleaned])
        
        # Predict probability
        proba = self.model.predict_proba(tfidf_vec)[0]
        phishing_prob = float(proba[1])
        legit_prob = float(proba[0])
        
        # If weights available, compute per-token contribution
        token_contributions = []
        if self.weights is not None:
            nonzero_indices = tfidf_vec.nonzero()[1]
            for idx in nonzero_indices:
                val = tfidf_vec[0, idx]
                w = self.weights[idx]
                contrib = val * w
                token = self.feature_names[idx]
                token_contributions.append({
                    "token": token,
                    "tfidf": round(float(val), 4),
                    "weight": round(float(w), 4),
                    "contribution": round(float(contrib), 4)
                })
                
        # Sort tokens by contribution to phishing vs legitimate
        phish_pushes = sorted(
            [t for t in token_contributions if t["contribution"] > 0],
            key=lambda x: x["contribution"],
            reverse=True
        )[:top_k]
        
        legit_pushes = sorted(
            [t for t in token_contributions if t["contribution"] < 0],
            key=lambda x: x["contribution"]
        )[:top_k]
        
        return {
            "phishing_probability": phishing_prob,
            "legitimate_probability": legit_prob,
            "prediction": "Phishing" if phishing_prob >= 0.5 else "Legitimate",
            "top_phishing_triggers": phish_pushes,
            "top_legitimate_indicators": legit_pushes,
            "total_active_features": len(token_contributions)
        }

    def explain_with_lime(self, raw_text: str, num_features: int = 10, num_samples: int = 400) -> Dict[str, Any]:
        """
        Generates local surrogate explanations using LIME (Local Interpretable Model-agnostic Explanations).
        Perturbs input text tokens to estimate local decision boundaries.
        """
        def predict_proba_fn(texts: List[str]):
            # Vectorize raw texts after cleaning
            cleaned_texts = [clean_text(t) for t in texts]
            vecs = self.vectorizer.transform(cleaned_texts)
            return self.model.predict_proba(vecs)
        
        # Run LIME explainer
        exp = self.lime_explainer.explain_instance(
            raw_text,
            predict_proba_fn,
            labels=[1],
            num_features=num_features,
            num_samples=num_samples
        )
        
        lime_list = exp.as_list(label=1)
        phishing_cues = []
        legit_cues = []
        
        for word, weight in lime_list:
            if weight > 0:
                phishing_cues.append({"word": word, "weight": round(float(weight), 4)})
            else:
                legit_cues.append({"word": word, "weight": round(float(weight), 4)})
                
        return {
            "lime_features": lime_list,
            "phishing_cues": phishing_cues,
            "legitimate_cues": legit_cues,
            "intercept": float(exp.intercept[1]) if hasattr(exp, 'intercept') else 0.0
        }


if __name__ == "__main__":
    explainer = EmailExplainer()
    sample = """
    URGENT NOTICE! We noticed unauthorized access to your account from an unknown IP address.
    Your online banking access has been suspended. Please click here to verify your credentials immediately.
    http://secure-update-bank.xyz/verify
    """
    print("--- MATHEMATICAL TF-IDF ATTRIBUTION ---")
    res = explainer.explain_instance_weights(sample)
    print("Prediction:", res["prediction"], f"({res['phishing_probability']*100:.1f}%)")
    print("Top Phishing Triggers:")
    for t in res["top_phishing_triggers"][:5]:
        print(f"  + {t['token']:<20} contrib={t['contribution']:+.4f} (tfidf={t['tfidf']}, weight={t['weight']})")
        
    print("\n--- LIME EXPLAINER ---")
    lime_res = explainer.explain_with_lime(sample, num_samples=150)
    print("Top LIME Phishing Cues:", lime_res["phishing_cues"][:5])
