# Sentiment Analysis Hyperparameter Tuning

A deep learning project implementing and comparing RNN, LSTM, and GRU architectures for sentiment analysis on Indonesian mobile banking app reviews, optimized using Optuna hyperparameter tuning.

## 📖 About This Project

This project was developed as a comprehensive assignment to explore the impact of hyperparameter tuning in Natural Language Processing (NLP). By utilizing a dataset of ~12,000 Indonesian mobile banking app reviews categorized into positive, negative, and neutral sentiments, this repository benchmarks the performance of **SimpleRNN**, **LSTM**, and **GRU** models.

Instead of manual trial-and-error or exhaustive grid search, the project leverages **Optuna** (Tree-structured Parzen Estimator) to dynamically search for the best hyperparameter combinations. 

### 🎯 Key Features
- **Automated Notebook Generation**: A robust python script (`src/generate_notebook.py`) that scaffolds the entire Jupyter Notebook, ensuring standardization and reproducibility.
- **Dynamic "Blue Sky" EDA**: Visualizations dynamically map colors to data values (higher values = darker blue, lower values = lighter blue) adhering to specific aesthetic philosophies.
- **Automated Narrative Analysis**: The notebook automatically drafts a narrative conclusion detailing the best hyperparameters, the reasoning behind them, and architecture comparisons based on the tuning results.
- **Structured Logging**: All tuning and experiment logs are safely written to a dedicated `logs/` directory to keep the notebook output clean.

---

## 📁 Project Structure

```text
├── app/                  # Application source code (if any)
├── config/               # Configuration files
├── data/
│   ├── external/         # External data sources
│   ├── processed/        # Processed data and CSV outputs (e.g., Tabel 2, Evaluasi Test Set)
│   └── raw/              # Raw dataset (dataset_perbankan.csv goes here)
├── docs/                 # Assignment rubrics, documentation, and final PDF reports
├── images/
│   └── output/           # Output visualizations and graphs for reporting
├── logs/                 # Execution logs (experiment.log, optuna_study.log)
├── models/               # Saved best model files (.keras)
├── notebooks/            # Generated Jupyter Notebooks
├── src/                  # Source scripts (generate_notebook.py)
├── test/                 # Testing and verification scripts
└── README.md             # Project documentation (this file)
```

---

## ⚙️ Hyperparameter Tuning Strategy

The tuning process searches for the optimal configuration over **30 trials per architecture**, maximizing the **Macro F1-Score** on a stratified validation set.

| Parameter | Search Space / Values | Strategy |
|:----------|:----------------------|:---------|
| **Learning Rate** | `[1e-4, 1e-2]` | Log-uniform |
| **Batch Size** | `[16, 32, 64]` | Categorical |
| **Hidden Units** | `[32, 64, 128]` | Categorical |
| **Dropout Rate** | `[0.1, 0.5]` | Uniform |
| Optimizer | Adam | Static |
| Loss Function | Sparse Categorical Crossentropy | Static |
| Epochs | 20 (with Early Stopping patience=3) | Static |

---

## 🚀 How to Run

### 1. Environment Setup
Make sure you have Python 3.11+ installed. Set up the virtual environment:
```powershell
# Create virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare the Dataset
Ensure the raw dataset is placed in the raw data folder:
`data/raw/dataset_perbankan.csv`

### 3. Generate the Notebook
Generate the full 43-cell Jupyter Notebook automatically by running:
```powershell
python src/generate_notebook.py
```
This will create `notebooks/hyperparameter_tuning_analysis_sentiment.ipynb`.

### 4. Run the Experiment
Launch Jupyter Notebook and execute all cells:
```powershell
jupyter notebook notebooks/hyperparameter_tuning_analysis_sentiment.ipynb
```
Select the **Python (Sentiment Tuning)** kernel. The notebook will process the data, tune the models, and output all tables (`data/processed/`), images (`images/output/`), logs (`logs/`), and the best models (`models/`).

---

## 📊 Evaluation & Reporting
Upon completion of the notebook, you will find:
1. **Analisis Naratif Otomatis (Section 6.7)**: A dynamically drafted text block answering why specific hyperparameters performed well.
2. **Optuna Parameter Importance Plots**: Visual guides showing which hyperparameters impacted the F1-Score the most.
3. **Confusion Matrices & Classification Reports**: For final Test Set evaluation.

You can directly copy the text, tables, and images into your final PDF assignment report.

*Developed with TensorFlow, Keras, Optuna, and Seaborn.*
