from preprocessing import load_and_preprocess_data
from feature_engineering import vectorize_text
from model import train_model

# Path to your dataset
dataset_path = '/path/to/your/imdb_dataset.csv'

# Load and preprocess the data
df = load_and_preprocess_data(dataset_path)

# Vectorize the text data
X, vectorizer = vectorize_text(df, 'review')  # Assuming 'review' is your text column

# Train and evaluate the model
train_model(X, df['sentiment'])  # Assuming 'sentiment' is your label column
