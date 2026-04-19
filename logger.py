import csv
import os
import datetime
import pandas as pd

# Use absolute path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, 'data', 'predictions.csv')

def initialize_log():
    """Creates the CSV file with headers if it doesn't exist or is empty."""
    is_empty = not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0
    if is_empty:
        headers = ['timestamp', 'f0', 'f1', 'f2', 'f3', 'f4', 'prediction', 'confidence', 'model_version', 'ground_truth']
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"Initialized log file at {LOG_FILE}")

def log_prediction(features, prediction, confidence, model_version, ground_truth=None):
    """
    Logs a single prediction to the CSV file.
    """
    initialize_log()
    timestamp = datetime.datetime.now().isoformat()
    # Ensure we strictly have 5 features to match the CSV schema [f0, f1, f2, f3, f4]
    features_fixed = (list(features) + [0.0]*5)[:5]
    row = [timestamp] + features_fixed + [prediction, confidence, model_version, ground_truth]
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def get_recent_logs(limit=50):
    """Reads the most recent logs as a pandas DataFrame."""
    if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(LOG_FILE).tail(limit)
    except Exception as e:
        print(f"Error reading logs: {e}")
        return pd.DataFrame()
