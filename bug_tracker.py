# bug_tracker.py

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify

nltk.download('stopwords')

# Step 1: Mock Dataset
data = {
    'description': [
        "Login button doesn't work", 
        "Database connection timeout", 
        "UI is unresponsive on mobile", 
        "Backend API throws 500 error"
    ],
    'tag': ['UI', 'Database', 'UI', 'Backend'],
    'priority': ['High', 'Medium', 'High', 'Critical']
}

df = pd.DataFrame(data)

# Step 2: Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['processed_description'] = df['description'].apply(preprocess_text)

# Encode labels
tag_mapping = {tag: idx for idx, tag in enumerate(df['tag'].unique())}
priority_mapping = {priority: idx for idx, priority in enumerate(df['priority'].unique())}
df['tag_encoded'] = df['tag'].map(tag_mapping)
df['priority_encoded'] = df['priority'].map(priority_mapping)

# Step 3: Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_description'])
y_tag = df['tag_encoded']
y_priority = df['priority_encoded']

# Train-test split
X_train, X_test, y_tag_train, y_tag_test = train_test_split(X, y_tag, test_size=0.2, random_state=42)
_, _, y_priority_train, y_priority_test = train_test_split(X, y_priority, test_size=0.2, random_state=42)

# Step 4: Train Models
tag_model = RandomForestClassifier()
tag_model.fit(X_train, y_tag_train)

priority_model = RandomForestClassifier()
priority_model.fit(X_train, y_priority_train)

# Step 5: Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    description = preprocess_text(data['description'])
    vectorized_description = vectorizer.transform([description])

    # Predict tag and priority
    predicted_tag = tag_model.predict(vectorized_description)[0]
    predicted_priority = priority_model.predict(vectorized_description)[0]

    return jsonify({
        'predicted_tag': list(tag_mapping.keys())[list(tag_mapping.values()).index(predicted_tag)],
        'predicted_priority': list(priority_mapping.keys())[list(priority_mapping.values()).index(predicted_priority)]
    })

if __name__ == "__main__":
    app.run(debug=True)
