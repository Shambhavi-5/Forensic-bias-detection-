from sentence_transformers import SentenceTransformer, util
import torch

class EmbeddingModel:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the sentence transformer model for generating embeddings.
        Using a smaller, faster model so it runs fine on CPU.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, sentences: list):
        """Generates embeddings for a list of sentences."""
        return self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)

    def compute_similarity(self, query_embedding, corpus_embeddings):
        """Computes cosine similarity between a query and a corpus of embeddings."""
        return util.cos_sim(query_embedding, corpus_embeddings)

# Singleton instance for simple import
_model_instance = None

def get_embedding_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = EmbeddingModel()
    return _model_instance
