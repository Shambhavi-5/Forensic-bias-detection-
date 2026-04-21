from embedding_model import get_embedding_model
import torch

class EvidenceRetriever:
    def __init__(self):
        self.embedder = get_embedding_model()

    def retrieve(self, claim_text: str, document_sentences: list, top_k: int = 3):
        """
        Retrieves the top_k sentences from the document that are semantically related to the claim.
        To avoid retrieving the claim itself or exact rephrasings, we might do a simple filter,
        but for baseline NLI, highly similar sentences are best.
        """
        if not document_sentences:
            return []
            
        # Encode claim and corpus
        # We can optimize this by pre-computing document embeddings in the main loop
        pass

    def retrieve_with_precomputed(self, claim_idx: int, claim_text: str, document_sentences: list, document_embeddings, top_k: int = 3):
        """
        Retrieves the top_k sentences using pre-computed embeddings.
        This is much faster for document-level inference.
        """
        claim_emb = document_embeddings[claim_idx]
        
        # Calculate cosine similarity
        cos_scores = self.embedder.compute_similarity(claim_emb, document_embeddings)[0]
        
        # We don't want the claim itself to be its own evidence
        cos_scores[claim_idx] = -1.0 
        
        # Get top-k scores
        top_results = torch.topk(cos_scores, k=min(top_k, len(document_sentences)))
        
        retrieved_evidence = []
        for score, idx in zip(top_results[0], top_results[1]):
            # Filter out sentences that are identically the claim or trivially close
            if score > 0.95:
                continue
                
            retrieved_evidence.append({
                "index": int(idx),
                "text": document_sentences[idx],
                "similarity_score": float(score)
            })
            
        # If we filtered out too many, we just take the remaining best
        return retrieved_evidence[:top_k]

# Singleton instance
_retriever_instance = None

def get_evidence_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = EvidenceRetriever()
    return _retriever_instance
