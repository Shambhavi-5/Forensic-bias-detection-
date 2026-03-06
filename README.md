# AI-Assisted Cognitive Bias Risk Detection

This repository contains the prototype for an AI-assisted pipeline that detects cognitive bias risks (such as confirmation bias, anchoring bias, and overconfidence) in forensic and legal reports. 

## Features
- **Text Extraction:** Extracts raw text from PDF and DOCX reports.
- **NLP Processing:** Utilizes NLTK to evaluate sentence-level linguistic features (absolutes, subjective phrasing, hedging).
- **Machine Learning Inference:** A pre-trained Random Forest model that predicts Bias Risk Classification (Low, Medium, High).
- **Web UI:** A Streamlit dashboard for drag-and-drop document analysis and interactive reporting.

## Setup Instructions

1. Install the required dependencies:
```bash
pip install pandas scikit-learn nltk streamlit pdfplumber python-docx
```

2. Download the NLTK packages (if not already downloaded):
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
```

3. Run the web interface:
```bash
streamlit run app.py
```

## Repository Structure
- `app.py`: The Streamlit web application.
- `extract_text.py`: Module for handling PDF and DOCX file reading.
- `dataset_prep.py`: Logic for NLTK sentence tokenization and bias feature extraction.
- `predict_bias.py`: Inference engine using the ML model.
- `generate_report.py`: Command-line tool to generate a full CSV report for a document.
- `model_training.py`: Script used to train the Random Forest text classifier.
