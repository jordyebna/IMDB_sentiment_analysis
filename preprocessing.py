import pandas as pd

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Add your preprocessing steps here
    # For example, cleaning text, handling missing values, etc.

    return df
