import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from preprocessing import preprocess

MBERT_MODEL = 'bert-base-multilingual-cased'

def load_models():
    ensemble = joblib.load('models/ensemble_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    tokenizer = AutoTokenizer.from_pretrained(MBERT_MODEL)
    model = AutoModel.from_pretrained(MBERT_MODEL)
    return ensemble, tfidf, tokenizer, model

def predict(text, lang_code='hi'):
    ensemble, tfidf, tokenizer, model = load_models()
    
    text = preprocess(text, lang_code)
    
    inputs = tokenizer(text, return_tensors='pt',
                      truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    mbert_feat = outputs.last_hidden_state[:, 0, :].numpy()
    
    tfidf_feat = tfidf.transform([text]).toarray()
    
    X = np.hstack([mbert_feat, tfidf_feat])
    
    prob = ensemble.predict_proba(X)[0][1]
    label = 'FAKE' if prob >= 0.5 else 'REAL'
    
    return label, round(prob, 4)

if __name__ == '__main__':
    text = input("Enter news text: ")
    lang = input("Enter language code (hi/bn/or/ta/te): ")
    label, confidence = predict(text, lang)
    print(f"Prediction: {label} (Confidence: {confidence})")
