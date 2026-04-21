"""
nli_verifier.py — NLI-based claim-evidence verification using a dedicated MNLI model.

Uses CrossEncoder-style inference or a HuggingFace text-classification pipeline
pre-trained on the MNLI task to judge whether evidence entails, contradicts, or
is neutral to the claim being verified.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Label IDs for models trained on MNLI (order matters)
# facebook/bart-large-mnli: LABEL_0=contradiction, LABEL_1=neutral, LABEL_2=entailment
# roberta-large-mnli:       LABEL_0=contradiction, LABEL_1=neutral, LABEL_2=entailment
# distilbert-base-uncased-finetuned-sst-2-english: NOT for NLI
MNLI_LABEL_MAP = {
    "LABEL_0": "contradiction",
    "LABEL_1": "neutral",
    "LABEL_2": "entailment",
    # also handle explicit named labels from different model variants
    "entailment": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "CONTRADICTION": "contradiction",
    "NEUTRAL": "neutral",
    "ENTAILMENT": "entailment",
}


class NLIVerifier:
    def __init__(self, model_name: str = "cross-encoder/nli-distilroberta-base"):
        """
        Initializes an NLI pipeline using a dedicated cross-encoder NLI model.
        'cross-encoder/nli-distilroberta-base' is specifically fine-tuned for 
        NLI sentence-pair classification: contradiction / entailment / neutral.
        
        It is small enough to run on CPU quickly.
        """
        self.model_name = model_name
        device = 0 if torch.cuda.is_available() else -1

        # Using the text-classification pipeline — this model expects [sentence1, sentence2]
        # and returns entailment/neutral/contradiction
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            top_k=None,   # Return all label scores (not just the top one)
        )

    def verify(self, claim: str, evidence_list: list) -> dict:
        """
        Verifies the claim against each piece of retrieved evidence via NLI.

        Args:
            claim:         The claim sentence (hypothesis).
            evidence_list: List of evidence dicts {'text', 'similarity_score', 'index'}.

        Returns:
            {
                "support_label":  "Supported" | "Contradicted" | "Neutral/Unsupported",
                "support_score":  float,
                "details":        list of per-evidence NLI dicts
            }
        """
        if not evidence_list:
            return {
                "support_label": "No Evidence",
                "support_score": 0.0,
                "details": []
            }

        details = []
        for ev in evidence_list:
            premise    = ev["text"]
            hypothesis = claim

            try:
                # Pipeline accepts (text, text_pair) for NLI sentence-pair models
                raw_results = self.pipe(premise, text_pair=hypothesis)

                # raw_results is a list of [{"label": ..., "score": ...}, ...]
                # Normalise labels to lowercase entailment/neutral/contradiction
                label_scores = {}
                for item in raw_results:
                    norm = MNLI_LABEL_MAP.get(item["label"], item["label"].lower())
                    label_scores[norm] = item["score"]

                # Determine verdict for this specific evidence sentence
                top_nli_label = max(label_scores, key=label_scores.get)
                top_nli_score = label_scores[top_nli_label]

                # Normalise to user-facing label
                if top_nli_label == "entailment":
                    user_label = "supported"
                elif top_nli_label == "contradiction":
                    user_label = "contradicted"
                else:
                    user_label = "neutral"

                details.append({
                    "evidence_text":   premise,
                    "nli_label":       user_label,
                    "nli_raw_label":   top_nli_label,
                    "confidence":      float(top_nli_score),
                    "all_scores":      label_scores,
                    "similarity":      ev.get("similarity_score", 0.0),
                })

            except Exception as e:
                # Fallback: mark as neutral so we don't crash the pipeline
                details.append({
                    "evidence_text":   premise,
                    "nli_label":       "neutral",
                    "nli_raw_label":   "error",
                    "confidence":      0.33,
                    "all_scores":      {},
                    "similarity":      ev.get("similarity_score", 0.0),
                    "error":           str(e),
                })

        # ── Aggregate verdict across all evidence pieces ──────────────────────
        # Strategy:
        #   1. If ANY evidence strongly entails → Supported
        #   2. Else if ANY evidence strongly contradicts → Contradicted
        #   3. Otherwise → Neutral/Unsupported
        STRONG_THRESHOLD = 0.55

        entailed      = [d for d in details if d["nli_label"] == "supported"     and d["confidence"] >= STRONG_THRESHOLD]
        contradicted  = [d for d in details if d["nli_label"] == "contradicted"  and d["confidence"] >= STRONG_THRESHOLD]

        if entailed and not contradicted:
            overall_label = "Supported"
            overall_score = max(d["confidence"] for d in entailed)
        elif contradicted:
            overall_label = "Contradicted"
            overall_score = max(d["confidence"] for d in contradicted)
        else:
            overall_label = "Neutral/Unsupported"
            # Give a score proportional to best neutral/entailment score seen
            overall_score = max(d["confidence"] for d in details) if details else 0.5

        return {
            "support_label": overall_label,
            "support_score": float(overall_score),
            "details": details,
        }


# ── Singleton accessor ────────────────────────────────────────────────────────
_nli_instance = None

def get_nli_verifier() -> NLIVerifier:
    global _nli_instance
    if _nli_instance is None:
        _nli_instance = NLIVerifier()
    return _nli_instance
