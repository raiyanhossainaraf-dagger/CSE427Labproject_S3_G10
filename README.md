# Evidence-Aware Multi-Agent Retrieval-Augmented Generation for Scientific Literature Review

## Project Overview
This project investigates evidence-aware Retrieval-Augmented Generation (RAG) for scientific literature reviews. It employs a multi-agent system to ensure that generated claims are supported by verifiable evidence from research papers.

## Current Milestone: Milestone 1
**Objective:** Initial Exploratory Data Analysis (EDA) and Preprocessing.

## Project Structure
```text
CSE427_Evidence_Aware_RAG/
│
├── notebooks/
│   └── CSE427_Milestone1_EDA_Preprocessing.ipynb  # Primary executable notebook
│
├── data/
│   ├── raw/         # Original dataset files
│   ├── interim/     # Intermediate processing files
│   └── processed/   # Final cleaned and chunked data
│
├── outputs/
│   ├── figures/     # EDA visualizations
│   ├── tables/      # Statistical tables
│   └── summaries/   # Dataset and preprocessing summaries
│
├── src/             # Reusable Python modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda_utils.py
│   └── utils.py
│
├── report/
│   └── milestone1_report_notes.md  # Academic draft for the report
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup Instructions

### Google Colab (Recommended)
1. Upload the `notebooks/CSE427_Milestone1_EDA_Preprocessing.ipynb` to Google Colab.
2. The notebook will automatically install dependencies and download the QASPER dataset.
3. It creates the required directory structure in the Colab environment.

### Local Development (PyCharm/VS Code)
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook using your IDE's Jupyter integration.

## Dataset Information
- **Primary Dataset:** QASPER (Question Answering on Scientific Papers).
- **Secondary Dataset:** SciReviewGen (Documented for future use).

### QASPER Loading
The notebook uses a manual fallback loader because the official Hugging Face dataset script is deprecated. 
- It attempts to load `qasper-train-v0.3.json`, `qasper-dev-v0.3.json`, and `qasper-test-v0.3.json` from `data/raw/qasper_v0.3/`.
- If missing, it extracts them from `train_dev.tgz` and `test.tgz` in `data/raw/`.
- If archives are also missing, it downloads them from the official Amazon S3 buckets provided by AllenAI.

## License
Academic project for CSE427.
