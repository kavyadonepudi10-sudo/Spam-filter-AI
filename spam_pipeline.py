import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve

print("1. Loading dataset...")
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

df['target'] = df['label'].map({'spam': 1, 'ham': 0})
print(f"Dataset loaded successfully. Total messages: {len(df)}")

print("2. Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['target'], test_size=0.2, random_state=42, stratify=df['target']
)

print("3. Converting text to features using TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("4. Training and comparing models...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_tfidf, y_train)

for name, model in [("Naive Bayes", nb_model), ("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
    preds = model.predict(X_test_tfidf)
    print(f"\n--- {name} Performance ---")
    print(classification_report(y_test, preds))

print("5. Tuning decision threshold for Logistic Regression...")
y_scores = lr_model.predict_proba(X_test_tfidf)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

target_precision = 0.90
valid_indices = np.where(precisions >= target_precision)[0]
optimal_threshold = thresholds[valid_indices[0]] if len(valid_indices) > 0 else 0.5
print(f"Optimal Threshold chosen (Precision >= {target_precision}): {optimal_threshold:.4f}")

print("6. Packaging as a deployment function...")
def predict(new_message: str):
    """Predicts whether a new message is spam or not spam."""
    msg_tfidf = vectorizer.transform([new_message])
    prob_spam = lr_model.predict_proba(msg_tfidf)[0][1]
    is_spam = prob_spam >= optimal_threshold
    
    return {
        "message": new_message,
        "spam_probability": round(float(prob_spam), 4),
        "prediction": "spam" if is_spam else "not spam",
        "threshold_used": round(float(optimal_threshold), 4)
    }

print("\n--- Testing Live Predictions ---")
print(predict("WINNER! You have won a free $1,000 Walmart gift card. Click here now to claim."))
print(predict("Hey, are we still meeting up for coffee tomorrow afternoon?"))