import gc
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def train_and_eval_all_models(X_train, X_test, y_train, y_test, preprocessor, config: dict):
    """Trains baseline and tree models within leak-free pipelines."""
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=500, class_weight='balanced', random_state=config['model']['random_state']
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=config['model']['rf_n_estimators'], max_depth=8, 
            class_weight='balanced', n_jobs=1, random_state=config['model']['random_state']
        ),
        "XGBoost": XGBClassifier(
            n_estimators=30, max_depth=4, n_jobs=1, eval_metric='logloss', 
            random_state=config['model']['random_state']
        )
    }
    
    results = []
    fitted_pipelines = {}
    
    for name, model in models.items():
        # Combine Preprocessor + Estimator into a single leak-free Pipeline
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        full_pipeline.fit(X_train, y_train)
        preds = full_pipeline.predict(X_test)
        probs = full_pipeline.predict_proba(X_test)[:, 1]
        
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Recall (Sensitivity)": recall_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, probs)
        })
        fitted_pipelines[name] = full_pipeline
        gc.collect()
        
    return pd.DataFrame(results), fitted_pipelines