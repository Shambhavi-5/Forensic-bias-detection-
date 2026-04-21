"""
main.py — End-to-end pipeline orchestrator for Legal Bias & Reasoning Flaw Detection.

Pipeline flow:
    PDF → pdf_parser → embedding_model → claim_classifier → evidence_retriever → nli_verifier → reasoning_engine → CSV
"""

import argparse
import sys
import os
import pandas as pd
from tqdm import tqdm

from pdf_parser import extract_sentences
from embedding_model import get_embedding_model
from claim_classifier import get_claim_classifier
from evidence_retriever import get_evidence_retriever
from nli_verifier import get_nli_verifier
from reasoning_engine import get_reasoning_engine

# Labels from claim_classifier that we treat as "Claims" and apply NLI verification to
CLAIM_LABELS = {"Claim or Conclusion", "Legal Argument"}

def run_pipeline(file_path: str, top_k: int = 3, output_csv: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Runs the full bias detection pipeline on the given document.

    Args:
        file_path:  Path to the legal/forensic PDF, DOCX, or TXT file.
        top_k:      Number of evidence sentences to retrieve per claim.
        output_csv: Optional path to save the output CSV.
        verbose:    Print progress to stdout.

    Returns:
        A pandas DataFrame with the full analysis results.
    """
    if verbose:
        print(f"\n[1/6] Extracting and segmenting text from: {file_path}")
    sentences = extract_sentences(file_path)
    if not sentences:
        print("ERROR: No valid sentences were extracted from the document.")
        return pd.DataFrame()
    if verbose:
        print(f"      >> {len(sentences)} sentences extracted.")

    # --------------------------------------------------------
    # Step 2: Generate embeddings for all sentences ONCE
    # --------------------------------------------------------
    if verbose:
        print("[2/6] Generating sentence embeddings...")
    embedder = get_embedding_model()
    document_embeddings = embedder.encode(sentences)
    if verbose:
        print(f"      >> Embeddings shape: {document_embeddings.shape}")

    # --------------------------------------------------------
    # Step 3: Classify every sentence
    # --------------------------------------------------------
    if verbose:
        print("[3/6] Classifying sentences (zero-shot: Claim / Evidence / Argument / Background)...")
    classifier = get_claim_classifier()
    classified = classifier.classify(sentences)

    # --------------------------------------------------------
    # Step 4 + 5: For each Claim, retrieve evidence and run NLI
    # --------------------------------------------------------
    if verbose:
        print("[4/6] Retrieving evidence & running NLI verification for claims...")
    retriever = get_evidence_retriever()
    nli = get_nli_verifier()
    engine = get_reasoning_engine()

    results = []
    claim_count = sum(1 for c in classified if c["label"] in CLAIM_LABELS)
    if verbose:
        print(f"      >> {claim_count} sentences identified as Claims/Arguments.")

    iterator = tqdm(enumerate(classified), total=len(classified), desc="Analyzing", disable=not verbose)
    for idx, sent_obj in iterator:
        label = sent_obj["label"]

        if label in CLAIM_LABELS:
            # Retrieve top-k semantically similar sentences as candidate evidence
            evidence = retriever.retrieve_with_precomputed(
                claim_idx=idx,
                claim_text=sent_obj["text"],
                document_sentences=sentences,
                document_embeddings=document_embeddings,
                top_k=top_k
            )

            # Run NLI: does the evidence entail, contradict, or neither support the claim?
            nli_result = nli.verify(sent_obj["text"], evidence)

            # Compute bias risk and build the result row
            evaluation = engine.evaluate_claim(sent_obj, nli_result)

            # Build the full evidence block string
            evidence_texts = " | ".join(
                [f'[{e["similarity_score"]:.2f}] {e["text"][:120]}' for e in evidence]
            )

            results.append({
                "Sentence":                evaluation["sentence"],
                "Sentence_Type":           label,
                "Classifier_Confidence":   round(sent_obj["confidence"], 3),
                "Retrieved_Evidence":      evidence_texts,
                "NLI_Verdict":             nli_result["support_label"],
                "NLI_Score":               round(nli_result["support_score"], 3),
                "Bias_Risk_Score":         evaluation["bias_risk_score"],
                "Final_Label":             evaluation["final_label"],
                "Explanation":             evaluation["explanation"],
            })
        else:
            # Non-claim sentences: just log them with no risk
            results.append({
                "Sentence":                sent_obj["text"],
                "Sentence_Type":           label,
                "Classifier_Confidence":   round(sent_obj["confidence"], 3),
                "Retrieved_Evidence":      "N/A",
                "NLI_Verdict":             "N/A",
                "NLI_Score":               0.0,
                "Bias_Risk_Score":         0.0,
                "Final_Label":             "Background / Evidence",
                "Explanation":             "Sentence classified as non-claim; not subject to NLI bias analysis.",
            })

    # --------------------------------------------------------
    # Step 6: Optional graph-based reasoning flow analysis
    # --------------------------------------------------------
    if verbose:
        print("[5/6] Building semantic graph for reasoning flow analysis...")
    graph_info = engine.build_document_graph(sentences, document_embeddings)
    if verbose:
        print(f"      >> Graph: {graph_info['num_components']} connected components.")
        print(f"      >> Isolated (chain-less) sentences: {len(graph_info['isolated_sentence_indices'])}")

    # Mark isolated sentences as having a potential missing reasoning chain
    isolated_idxs = set(graph_info["isolated_sentence_indices"])
    for i, r in enumerate(results):
        if i in isolated_idxs and r["Final_Label"] not in ("Background / Evidence",):
            r["Explanation"] += " [NOTE: Sentence is isolated in the semantic graph — no strong reasoning chain detected.]"

    # --------------------------------------------------------
    # Build DataFrame and save
    # --------------------------------------------------------
    df = pd.DataFrame(results)

    # Sort: most risky first
    label_order = {
        "Potentially Biased / Flawed": 4,
        "Weakly Supported (Overconfident)": 3,
        "Weakly Supported": 2,
        "Inconclusive": 1,
        "Well-Reasoned": 0,
        "Background / Evidence": -1,
    }
    df["_sort"] = df["Final_Label"].map(label_order).fillna(0)
    df = df.sort_values(["_sort", "Bias_Risk_Score"], ascending=[False, False]).drop(columns=["_sort"]).reset_index(drop=True)

    if output_csv:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        if verbose:
            print(f"[6/6] Results saved to: {output_csv}")

    return df


def print_summary(df: pd.DataFrame):
    """Prints a readable summary of the pipeline results to stdout."""
    print("\n" + "="*70)
    print("  LEGAL BIAS & REASONING FLAW DETECTION — SUMMARY REPORT")
    print("="*70)
    total = len(df)
    claims = df[df["Sentence_Type"].isin(CLAIM_LABELS)]
    print(f"\n  Total Sentences Analyzed : {total}")
    print(f"  Claims / Arguments Found : {len(claims)}")
    label_counts = df["Final_Label"].value_counts()
    print("\n  Verdict Distribution:")
    for label, count in label_counts.items():
        print(f"    {label:<40} {count}")
    print("\n  Top 5 Highest Risk Sentences:")
    print("-"*70)
    top = df[df["Bias_Risk_Score"] > 0].head(5)
    for _, row in top.iterrows():
        print(f"\n  [{row['Final_Label']}] (Risk={row['Bias_Risk_Score']}, NLI={row['NLI_Verdict']})")
        print(f"  SENTENCE : {row['Sentence'][:120]}...")
        print(f"  REASON   : {row['Explanation']}")
    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Legal Bias & Reasoning Flaw Detector — NLI-powered pipeline."
    )
    parser.add_argument("input_file", help="Path to the legal document (PDF, DOCX, or TXT).")
    parser.add_argument("--top_k", type=int, default=3, help="Number of evidence sentences to retrieve per claim (default: 3).")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path. Defaults to <input_file>_bias_report.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"ERROR: File not found: {args.input_file}")
        sys.exit(1)

    if args.output is None:
        base = os.path.splitext(args.input_file)[0]
        args.output = f"{base}_bias_report.csv"

    df = run_pipeline(
        file_path=args.input_file,
        top_k=args.top_k,
        output_csv=args.output,
        verbose=True
    )

    if not df.empty:
        print_summary(df)
