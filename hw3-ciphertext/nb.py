from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from cipher_data import CipherTxtData
from utils import accuracy_score_scalers

def fit(X, y, smoothing=1):
    nb = MultinomialNB(alpha=smoothing)
    nb.fit(X, y)

    return nb

def predict(nb, X):
    y_pred = nb.predict(X)

    return y_pred

def transform_data(train_data, dev_data):
    vectorizer = TfidfVectorizer(lowercase=False, binary=False)
    vectorizer.fit(train_data.X)

    X_train = vectorizer.transform(train_data.X)
    X_dev = vectorizer.transform(dev_data.X)

    return X_train, X_dev, train_data.y, dev_data.y

def main():
    train_data = CipherTxtData(mode="train", split=False)
    dev_data = CipherTxtData(mode="dev", split=False)

    X_train, X_dev, y_train, y_dev = transform_data(train_data, dev_data)

    print("\nPerformance on dev dataset:")
    for smoothing in range(1, 10):
        nb = fit(X_train, y_train, smoothing)
        y_pred = predict(nb, X_dev)
        score = accuracy_score_scalers(y_dev, y_pred)
        print(f"\tsmoothing: {smoothing} \tscore: {score}")

if __name__ == "__main__":
    print("Training multinomial naive bayes classifier.")
    main()
