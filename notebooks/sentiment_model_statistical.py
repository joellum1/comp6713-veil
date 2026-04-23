"""
    pipeline:
        text -> TF-IDF features -> SVM classifier -> sentiment prediction (accuracy, precision, recall, f1-score)
    1. Load data
    2. Split training and testing set
    3. Train the model
    4. Evaluate the model
    5. Predict the sentiment of the test set
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import os
import joblib

def main():
    # load data
    df = pd.read_csv("data/processed/sentiment/stitched_sentiment.csv")
    X_text = df["data"]
    y_label = df["sentiment"].apply(lambda x: 1 if x == "positive" else -1 if x == "negative" else 0)
    # split training and testing set
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y_label, test_size=0.2, random_state=42)
    # pre processing
    tfidf = TfidfVectorizer(max_features=10000)
    tfidf.fit(X_train_text)               # fit on train only
    X_train = tfidf.transform(X_train_text)
    X_test  = tfidf.transform(X_test_text)
    # train the model
    rbf = svm.SVC(kernel='rbf', gamma='scale', C=10, class_weight='balanced').fit(X_train, y_train)
    # evaluate the model
    pred_rbf = rbf.predict(X_test)
    accuracy_rbf = accuracy_score(y_test, pred_rbf) * 100
    precision = precision_score(y_test, pred_rbf, average='weighted')
    recall = recall_score(y_test, pred_rbf, average='weighted')
    f1 = f1_score(y_test, pred_rbf, average='weighted')
    # print the results
    print(f"RBF Kernel Accuracy: {accuracy_rbf:.2f}%")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    # save model
    os.makedirs("src/models/sentiment_statistical_model", exist_ok=True)
    joblib.dump(rbf, "src/models/sentiment_statistical_model/svm_model.pkl")
    joblib.dump(tfidf, "src/models/sentiment_statistical_model/tfidf_vectorizer.pkl")
    # confusion matrix
    cm = confusion_matrix(y_test, pred_rbf)
    # map numeric labels back to strings
    label_map = {-1: "negative", 0: "neutral", 1: "positive"}
    display_labels = [label_map[cls] for cls in rbf.classes_]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    # save the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Sentiment Model")
    plt.savefig("notebooks/report/statistical_model_confusion_matrix.png")
    print("\nConfusion matrix saved to notebooks/report/statistical_model_confusion_matrix.png")

if __name__ == "__main__":
    main()