import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModel
import torch
import joblib

MBERT_MODEL = 'bert-base-multilingual-cased'

def get_mbert_embeddings(texts, tokenizer, model, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', 
                          padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

def train(data_path):
    df = pd.read_csv(data_path)
    X_text = df['text'].tolist()
    y = df['label'].values

    tokenizer = AutoTokenizer.from_pretrained(MBERT_MODEL)
    model = AutoModel.from_pretrained(MBERT_MODEL)

    print("Extracting mBERT embeddings...")
    mbert_features = get_mbert_embeddings(X_text, tokenizer, model)

    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1,3), max_features=50000)
    tfidf_features = tfidf.fit_transform(X_text).toarray()

    X = np.hstack([mbert_features, tfidf_features])

    xgb = XGBClassifier(n_estimators=200, max_depth=6, 
                        learning_rate=0.1, scale_pos_weight=1.15)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, 
                               class_weight='balanced')
    meta = LogisticRegression(C=0.1, class_weight='balanced')

    ensemble = StackingClassifier(
        estimators=[('xgb', xgb), ('rf', rf)],
        final_estimator=meta,
        cv=5
    )

    print("Training ensemble...")
    ensemble.fit(X, y)
    joblib.dump(ensemble, 'models/ensemble_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    print("Done! Models saved.")

if __name__ == '__main__':
    train('data/dataset.csv')
