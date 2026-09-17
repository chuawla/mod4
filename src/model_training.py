import gc
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def train_and_eval_all_models(X_train, X_test, y_train, y_test, config: dict):
    # n_jobs=1 restricts CPU thread allocation to keep RAM stable
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=config['model']['random_state']),
        "Random Forest": RandomForestClassifier(n_estimators=config['model']['rf_n_estimators'], max_depth=8, n_jobs=1, random_state=config['model']['random_state']),
        "XGBoost": XGBClassifier(n_estimators=30, max_depth=4, n_jobs=1, eval_metric='logloss', random_state=config['model']['random_state'])
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
        
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Recall (Sensitivity)": recall_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, probs)
        })
        gc.collect()  # Release unused memory between model fits
        
    return pd.DataFrame(results), models