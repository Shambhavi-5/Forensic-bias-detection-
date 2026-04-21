from transformers import pipeline
import torch

class ClaimClassifier:
    def __init__(self, model_name: str = 'valhalla/distilbart-mnli-12-3'):
        """
        Initializes a zero-shot classification pipeline.
        We use a smaller distilled model for speed unless high-end GPU is available.
        """
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=device)
        self.candidate_labels = ["Claim or Conclusion", "Evidence or Fact", "Legal Argument", "Background or Procedure"]

    def classify(self, sentences: list) -> list:
        """
        Classifies sentences in batch.
        Returns a list of dicts with the highest scoring label for each sentence.
        """
        results = []
        # Process in chunks to avoid memory issues if doc is huge
        batch_size = 16
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i+batch_size]
            preds = self.classifier(batch, self.candidate_labels, multi_label=False)
            
            # Handle single sentence output edge case
            if isinstance(preds, dict):
                preds = [preds]
                
            for p in preds:
                top_label = p['labels'][0]
                top_score = p['scores'][0]
                results.append({
                    "text": p['sequence'],
                    "label": top_label,
                    "confidence": top_score
                })
        return results

# Singleton instance
_classifier_instance = None

def get_claim_classifier():
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ClaimClassifier()
    return _classifier_instance
