import os
import sys
from transformers import BertTokenizer, BertForSequenceClassification
from pprint import pprint
import torch
import joblib
import re
import unicodedata

# helper functions
def clean_text(text):
    # nicode normalisation -> ASCII where possible
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # strip basic HTML tags / entities
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    # expand common English contractions
    contractions = {
        r"won\'t": "will not", r"can\'t": "cannot", r"n\'t": " not",
        r"\'re": " are",      r"\'s":  " is",       r"\'d":  " would",
        r"\'ll": " will",     r"\'ve": " have",      r"\'m":  " am",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return (str(text)
                .replace("\n", " ")
                .replace("\r", " ")
                .replace("?", "")
                .strip()
            )

if __name__ == "__main__":
    quit = False
    choices = ["1", "2", "3", "4"]
    # load sentiment transformer model
    print("Loading models...")
    SENTIMENT_MODEL_DIR = "./src/models/finbert_model"
    sentiment_tokenizer = BertTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
    sentiment_model = BertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR)
    # load sentiment statistical model
    STATISTICAL_MODEL_DIR = "./src/models/sentiment_statistical_model"
    sentiment_svm_model = joblib.load(f"{STATISTICAL_MODEL_DIR}/svm_model.pkl")
    sentiment_tfidf_vectorizer = joblib.load(f"{STATISTICAL_MODEL_DIR}/tfidf_vectorizer.pkl")
    while not quit:
        print("\n" + "="*60)
        print("TESTING MENU")
        print("="*60)
        print("1. Test Statistical Model")
        print("2. Test Sentiment Model (FinBERT)")
        print("3. Test Summarization Model")
        print("4. Exit")
        print("="*60)
        choice = input("Enter your choice: ")
        if choice not in choices:
            print("Invalid choice. Please try again.")
            continue
        else:
            if choice == "4":
                quit = True
                continue
            # get input text
            text = input("Enter text to analyze: ")
            # clean text
            text = clean_text(text)            
            if choice == "1":
                # Transform input text using the saved vectorizer
                X = sentiment_tfidf_vectorizer.transform([text])
                prediction = sentiment_svm_model.predict(X)[0]
                # Map numeric label back to string
                label_map = {-1: "negative", 0: "neutral", 1: "positive"}
                label = label_map[prediction]
                # Get decision function scores for confidence indication
                scores = sentiment_svm_model.decision_function(X)[0]
                class_scores = {label_map[cls]: round(scores[i], 4) for i, cls in enumerate(sentiment_svm_model.classes_)}
                print(f"Label: {label}")
                print(f"Scores:")
                pprint(class_scores)
            elif choice == "2":
                sentiment_model.eval()
                inputs = sentiment_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                with torch.no_grad(): 
                    outputs = sentiment_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
                    # Map to labels (from your config.json)
                    id2label = {0: "positive", 1: "negative", 2: "neutral"}
                    predicted_id = probs.argmax().item()
                    result = {
                        "label": id2label[predicted_id],
                        "confidence": probs[predicted_id].item(),
                        "scores": {id2label[i]: probs[i].item() for i in range(3)}
                    }
                    pprint(result)
            else:
                pass
                # os.system("python src/test/test_lstm_model.py")
            
