# Deal-Origination-Analytics

**Deal Origination Blind Spots in Private Equity: Alternative Data Reveals Hidden PE Targets**

Author: Ayankhan Pathan (ESSEC Strategic Business Analytics)

This repository accompanies the ESSEC project investigating whether multi-signal alternative data can surface mid-market targets earlier than traditional origination channels. The materials are academic in style but designed with industry applicability: code, synthetic datasets, visualizations, and a LaTeX appendix suitable for inclusion in a project report.

## Contents
- `code/` : scripts to generate synthetic alt-data signals, build indices, and train a simple early-warning model.
- `data/` : synthetic datasets (signals + a small simulated deal pipeline).
- `images/` : pre-generated charts for inclusion in the report.
- `report/` : LaTeX appendix with figures (compile with pdflatex).
- `appendix/` : code snippets and tables used as appendices in the report.

## Quick start
1. Install Python dependencies:
```bash
pip install -r requirements.txt
python code/generate_visuals.py
pdflatex report/main.tex

## `requirements.txt`

numpy
pandas
matplotlib
scikit-learn
seaborn
requests


## `code/altdata_signals.py`
```python
"""altdata_signals.py
Generate synthetic alternative signals: hiring_velocity, employee_sentiment, web_traffic, search_interest.
Outputs CSV for analysis and plotting.
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def generate_signals(n=120, seed=2025):
    np.random.seed(seed)
    weeks = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='W')
    # baseline trends + shocks
    def gen(scale, drift):
        shocks = np.random.randn(n) * scale
        trend = np.linspace(0, drift, n)
        series = np.cumsum(shocks) + trend
        return series
    hiring = gen(0.8, 1.5) + np.where(np.arange(n)>80, -5, 0)  # late drop
    sentiment = gen(0.6, 0.8) + np.where(np.arange(n)>70, -3, 0)
    web = gen(1.0, 2.0) + np.where(np.arange(n)>60, -6, 0)
    search = gen(0.7, 1.0) + np.where(np.arange(n)>50, -4.5, 0)
    df = pd.DataFrame({'week': weeks, 'hiring_velocity': hiring, 'employee_sentiment': sentiment,
                       'web_traffic': web, 'search_interest': search})
    return df

def save_outputs(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    df = generate_signals()
    df.to_csv(path/'synthetic_alt_signals.csv', index=False)
    # simple plot
    plt.figure(figsize=(10,4))
    for col in ['hiring_velocity','employee_sentiment','web_traffic','search_interest']:
        plt.plot(df['week'], df[col], label=col, lw=1.5)
    plt.legend(loc='upper right', fontsize=8)
    plt.title('Synthetic Alternative Signals (illustrative)')
    plt.tight_layout()
    plt.savefig(path.parent/'images'/'alt_signals.png', dpi=150)
    print('Wrote', path/'synthetic_alt_signals.csv')

if __name__ == '__main__':
    save_outputs(Path(__file__).resolve().parents[1]/'data')


code/early_warning_model.py

"""early_warning_model.py
Constructs a Market Early-Signal Index and trains a simple classifier to predict a synthetic 'distress_event' occurring in future windows.
Improved label generation to ensure positive examples exist.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt

def build_index(df):
    signals = ['hiring_velocity','employee_sentiment','web_traffic','search_interest']
    # z-score normalize
    norm = (df[signals] - df[signals].mean()) / df[signals].std()
    df['early_signal_index'] = norm.mean(axis=1)
    return df

def generate_labels(df, lead=12):
    # label if index drops by threshold within lead window OR inject events at specific times
    idx = df['early_signal_index']
    future_min = idx.rolling(window=lead, min_periods=1).min().shift(-lead+1)
    # primary condition
    df['distress_event'] = (future_min < idx - 1.2).astype(int)
    # if too few positives, augment by marking 3 artificially created events at known drop windows
    if df['distress_event'].sum() < max(5, int(0.05*len(df))):
        n = len(df)
        candidates = [int(n*0.55), int(n*0.65), int(n*0.75)]
        for c in candidates:
            for i in range(max(0,c-2), min(n, c+3)):
                df.at[i, 'distress_event'] = 1
    return df

def train_model(df):
    features = ['early_signal_index']  # simple example
    X = df[features].fillna(0)
    y = df['distress_event'].fillna(0).astype(int)
    if y.nunique() < 2:
        print('Warning: not enough label variety, aborting training.')
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    print('AUC:', auc)
    print(classification_report(y_test, model.predict(X_test)))
    out = X_test.copy()
    out['pred_prob'] = preds
    out['label'] = y_test.values
    out.to_csv(Path(__file__).resolve().parents[1]/'data'/'early_warning_predictions.csv', index=False)
    plt.figure(figsize=(10,4))
    plt.plot(df['week'], df['early_signal_index'], label='Early Signal Index', lw=1.5)
    plt.scatter(df['week'], df['distress_event']* (df['early_signal_index'].min()-2), c='red', label='Distress Event (label)')
    plt.legend()
    plt.title('Early Signal Index and Distress Labels (illustrative)')
    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parents[1]/'images'/'early_warning_index.png', dpi=150)
    print('Saved predictions and figure.')

def main():
    data_path = Path(__file__).resolve().parents[1]/'data'/'synthetic_alt_signals.csv'
    df = pd.read_csv(data_path, parse_dates=['week'])
    df = build_index(df)
    df = generate_labels(df)
    df.to_csv(Path(__file__).resolve().parents[1]/'data'/'alt_signals_with_index.csv', index=False)
    train_model(df)

if __name__ == '__main__':
    main()

code/generate_visuals.py
"""generate_visuals.py
Runs altdata_signals and early_warning_model to create CSVs and PNGs.
"""
import subprocess, sys
subprocess.run([sys.executable, 'code/altdata_signals.py'])
subprocess.run([sys.executable, 'code/early_warning_model.py'])
print('Generated visuals and CSVs.')


report/main.tex

\documentclass[12pt,a4paper]{report}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{setspace}
\geometry{left=1in,right=1in,top=1in,bottom=1in}
\setstretch{1.25}
\begin{document}
\chapter*{Appendix: Deal Origination Visualizations}
\addcontentsline{toc}{chapter}{Appendix: Deal Origination Visualizations}

\begin{figure}[ht]
\centering
\includegraphics[width=0.9\textwidth]{../images/alt_signals.png}
\caption{Synthetic alternative data signals (hiring velocity, employee sentiment, web traffic, search interest).}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=0.9\textwidth]{../images/early_warning_index.png}
\caption{Early Signal Index and labeled distress events (illustrative).}
\end{figure}

\end{document}


appendix/appendix_code.tex

\chapter*{Appendix A: Code excerpts}
\addcontentsline{toc}{chapter}{Appendix A: Code excerpts}

\section*{Index construction (excerpt)}
\begin{verbatim}
norm = (df[signals] - df[signals].mean()) / df[signals].std()
df['early_signal_index'] = norm.mean(axis=1)
\end{verbatim}

\section*{Label generation (excerpt)}
\begin{verbatim}
future_min = idx.rolling(window=lead, min_periods=1).min().shift(-lead+1)
df['distress_event'] = (future_min < idx - 1.2).astype(int)
\end{verbatim}




