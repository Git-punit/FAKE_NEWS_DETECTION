import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data):
    X = data['text']
    y = data['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, vectorizer
