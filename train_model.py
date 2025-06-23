import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
cols = ['target','id','date','query','user','text']
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=cols)
df = df[df['target'] != 2]  # remove neutral
df['sentiment'] = df['target'].map({0: 'Negative', 4: 'Positive'})

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+|@\w+|[^A-Za-z\s]","", text)
    return text.lower().strip()

df['clean_text'] = df['text'].apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("✅ Model and vectorizer saved.")
