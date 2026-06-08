import joblib
import json

# Load the saved pipeline from 02_nlp_model_training.ipynb
pipeline = joblib.load("../models/nlp_tfidf_logreg_model.pkl")
tfidf = pipeline.named_steps['tfidf']
classifier = pipeline.named_steps['classifier']

# Extract TF-IDF parameters
vocab = tfidf.vocabulary_
idf = tfidf.idf_.tolist()

# Extract Logistic Regression parameters (MultiOutputClassifier contains a list of estimators)
coefs = [est.coef_[0].tolist() for est in classifier.estimators_]
intercepts = [est.intercept_[0].tolist() for est in classifier.estimators_]
labels = ["aggressive_tone", "hookup_focus", "negativity", "sarcasm_cynicism", "substance_risk", "incomplete_profile"]

model_data = {
    "vocab": vocab,
    "idf": idf,
    "coefs": coefs,
    "intercepts": intercepts,
    "labels": labels
}

with open("tfjs_model/nlp_model.json", "w") as f:
    json.dump(model_data, f)
print("NLP model exported to nlp_model.json")