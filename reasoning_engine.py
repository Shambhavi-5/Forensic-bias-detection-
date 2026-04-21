import networkx as nx

class ReasoningEngine:
    def __init__(self):
        # We can build a semantic graph to evaluate density and missing links
        pass

    def evaluate_claim(self, sentence_obj: dict, nli_result: dict) -> dict:
        """
        Evaluates a single claim to assign a bias risk score and a human-readable explanation.
        """
        # Feature extraction
        confidence = sentence_obj.get("confidence", 0.5)
        support_label = nli_result.get("support_label", "Neutral")
        
        # Calculate Risk Score
        # High Risk = High confidence claim but contradicted or completely unsupported
        risk_score = 0.0
        final_label = "Inconclusive"
        
        if support_label == "Supported":
            risk_score = 0.1
            final_label = "Well-Reasoned"
            explanation = "Claim is backed by logically entailed evidence within the document."
        elif support_label == "Contradicted":
            risk_score = 0.9 * confidence
            final_label = "Potentially Biased / Flawed"
            explanation = "Claim explicitly contradicts nearby evidence."
        else: # Neutral/Unsupported
            if confidence > 0.8:
                risk_score = 0.8
                final_label = "Weakly Supported (Overconfident)"
                explanation = "Claim expresses high certainty but lacks supporting evidence in the given context."
            else:
                risk_score = 0.4
                final_label = "Weakly Supported"
                explanation = "Claim lacks explicit supporting evidence but does not state absolute certainty."

        # Attach explanation and scores
        return {
            "sentence": sentence_obj.get("text", ""),
            "sentence_type": sentence_obj.get("label", "Claim"),
            "bias_risk_score": round(risk_score, 2),
            "final_label": final_label,
            "explanation": explanation,
            "evidence_used": [d["evidence_text"] for d in nli_result.get("details", [])]
        }

    def build_document_graph(self, sentences: list, embeddings):
        """
        Optional advanced module: Builds a networkx graph representing the semantic flow.
        (Implemented as a basic connectivity test)
        """
        import numpy as np
        from embedding_model import get_embedding_model
        
        embedder = get_embedding_model()
        sim_matrix = embedder.compute_similarity(embeddings, embeddings).cpu().numpy()
        
        G = nx.Graph()
        for i in range(len(sentences)):
            G.add_node(i, text=sentences[i])
            
        # Add edges for high semantic similarity (e.g. above 0.7)
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                if sim_matrix[i][j] > 0.7:
                    G.add_edge(i, j, weight=float(sim_matrix[i][j]))
        
        # We can find disconnected components to flag statements that have no reasoning chain
        components = list(nx.connected_components(G))
        isolated_nodes = [list(c)[0] for c in components if len(c) == 1]
        
        return {
            "num_components": len(components),
            "isolated_sentence_indices": isolated_nodes
        }

# Singleton instance
_reasoning_instance = None

def get_reasoning_engine():
    global _reasoning_instance
    if _reasoning_instance is None:
        _reasoning_instance = ReasoningEngine()
    return _reasoning_instance
