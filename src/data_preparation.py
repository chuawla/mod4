import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_and_optimize_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Clean Age
    if 'Age' in df.columns:
        df = df[df['Age'] > 0]
        
    # Map Survive target
    survive_map = {'1': 1, '1.0': 1, 'yes': 1, 'true': 1, '0': 0, '0.0': 0, 'no': 0, 'false': 0}
    if 'Survive' in df.columns:
        df['Survive'] = df['Survive'].astype(str).str.strip().str.lower().map(survive_map)

    # Clean Ejection Fraction
    if 'Ejection Fraction' in df.columns and df['Ejection Fraction'].dtype == 'O':
        df['Ejection Fraction'] = pd.to_numeric(
            df['Ejection Fraction'].astype(str).str.replace('%', '').str.strip(), 
            errors='coerce'
        )

    # Clean Binary Strings
    binary_map = {'yes': 1, 'no': 0, '1': 1, '0': 0, 'male': 1, 'm': 1, 'female': 0, 'f': 0}
    for col in ['Smoke', 'Diabetes', 'Gender']:
        if col in df.columns and df[col].dtype == 'O':
            df[col] = df[col].astype(str).str.strip().str.lower().map(binary_map)

    # Impute missing Creatinine
    if 'Creatinine' in df.columns:
        imputer = SimpleImputer(strategy='median')
        df['Creatinine'] = imputer.fit_transform(df[['Creatinine']])

    # Drop unused columns
    df = df.drop(columns=['ID', 'Favorite color'], errors='ignore')

    # Downcast floats and ints to save RAM
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int16)

    return df

def split_features_and_target(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)