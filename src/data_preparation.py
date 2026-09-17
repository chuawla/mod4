import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Loads CSV data, strips columns, and performs basic schema validation."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Assertions / Schema Checks
    required_cols = ['Survive', 'Age', 'Creatinine', 'Ejection Fraction']
    for col in required_cols:
        assert col in df.columns, f"Missing critical column: {col}"
    
    # Clean raw target & invalid ages
    df = df[df['Age'] > 0]
    survive_map = {'1': 1, '1.0': 1, 'yes': 1, 'true': 1, '0': 0, '0.0': 0, 'no': 0, 'false': 0}
    df['Survive'] = df['Survive'].astype(str).str.strip().str.lower().map(survive_map)
    
    # Clean numeric columns stored as strings
    if 'Ejection Fraction' in df.columns and df['Ejection Fraction'].dtype == 'O':
        df['Ejection Fraction'] = pd.to_numeric(
            df['Ejection Fraction'].astype(str).str.replace('%', '').str.strip(), errors='coerce'
        )
        
    df = df.drop(columns=['ID', 'Favorite color'], errors='ignore')
    return df

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Creates a ColumnTransformer to prevent data leakage during train/test splits."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    return ColumnTransformer([
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])

def split_data(df: pd.DataFrame, target_col: str, test_size=0.2, random_state=42):
    """Performs Stratified Train-Test Split to preserve class proportions."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)