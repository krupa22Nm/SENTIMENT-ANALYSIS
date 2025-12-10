import joblib
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, model_type='distilbert'):
        self.model_type = model_type
        if model_type == 'tfidf_svm':
            self.model = joblib.load('models/tfidf_svm.pkl')
        else:  # distilbert
            self.classifier = pipeline(
                "text-classification",
                model="models/distilbert_finetuned",
                tokenizer="models/distilbert_finetuned",
                return_all_scores=True
            )
    
    def predict(self, text):
        if self.model_type == 'tfidf_svm':
            pred = self.model.predict([text])[0]
            proba = max(self.model.predict_proba([text])[0])
            return {'label': pred, 'confidence': proba}
        else:
            results = self.classifier(text)[0]
            best = max(results, key=lambda x: x['score'])
            return {
                'label': best['label'].lower(),
                'confidence': best['score'],
                'all_scores': {r['label'].lower(): r['score'] for r in results}
            }

# Example usage
analyzer = SentimentAnalyzer(model_type='distilbert')
print(analyzer.predict("I love this phone! Battery lasts forever."))
# Output: {'label': 'positive', 'confidence': 0.987, ...}