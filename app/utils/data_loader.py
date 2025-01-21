import pandas as pd

def load_csv(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

