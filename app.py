from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model

def main():
    data = load_data('data/news.csv')
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(data)

    model = train_model(X_train, y_train)
    accuracy, report = evaluate_model(model, X_test, y_test)

    print("Fake News Detection Model Trained Successfully!\n")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:\n", report)

    # Test custom input
    test_news = ["Aliens have landed in Delhi and started a new government"]
    test_vector = vectorizer.transform(test_news)
    prediction = model.predict(test_vector)

    print("Test News:", test_news[0])
    print("Prediction:", "Real News" if prediction[0] == 1 else "Fake News")

if __name__ == "__main__":
    main()
