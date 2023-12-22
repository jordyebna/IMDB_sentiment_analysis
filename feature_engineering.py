from sklearn.feature_extraction.text import CountVectorizer
import joblib

def vectorize_text(df, text_column, max_features=1000):
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(df[text_column])

    # Optionally save the vectorizer
    joblib.dump(vectorizer, 'bow_vectorizer.joblib')

    return X, vectorizer