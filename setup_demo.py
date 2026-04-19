import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def create_demo_assets():
    print("Creating demo assets for AI Auditor...")
    
    # 1. Create a "Golden" Reference Dataset
    X = np.random.normal(0.5, 0.1, size=(200, 5))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    
    df = pd.DataFrame(X, columns=['f0', 'f1', 'f2', 'f3', 'f4'])
    df['target'] = y
    
    ref_path = os.path.join(MODELS_DIR, 'reference_data.csv')
    df.to_csv(ref_path, index=False)
    print(f"Created reference data at {ref_path}")
    
    # 2. Create Healthy Model
    model_h = RandomForestClassifier(n_estimators=10, random_state=42)
    model_h.fit(X, y)
    h_path = os.path.join(MODELS_DIR, 'healthy_model.pkl')
    joblib.dump(model_h, h_path)
    print(f"Created healthy model at {h_path}")
    
    # 3. Create Degraded Model (Biased - always predicts 1 if f0 > 0.4)
    model_d = RandomForestClassifier(n_estimators=10, random_state=42)
    # Train on biased data
    y_biased = (X[:, 0] > 0.4).astype(int)
    model_d.fit(X, y_biased)
    d_path = os.path.join(MODELS_DIR, 'degraded_model.pkl')
    joblib.dump(model_d, d_path)
    print(f"Created degraded model at {d_path}")

if __name__ == "__main__":
    create_demo_assets()
