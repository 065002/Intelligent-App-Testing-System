# 🔬 Intelligent App Testing System (IATS)

A data-driven QA intelligence platform that learns from historical bug data to predict defect risk, cluster bug patterns, and prioritize testing efforts for software releases.

---

## 🚀 Live Demo

Deploy on **Streamlit Cloud** — one click, no configuration required.

---

## 📋 Features

| Page | Description |
|------|-------------|
| 📋 Data Overview | Upload or auto-load dataset, explore schema, missing values, data types |
| 📊 Exploratory Analysis | Bug counts by module, severity/status distributions, version trends, top 10 bugs |
| ⚠️ Risk Scoring | Module risk scores using occurrence + severity + reopen rate; color-coded table + radar chart |
| 🤖 Bug Prediction Model | Gradient Boosting classifier predicting bug status; confusion matrix, feature importance, live inference |
| 🔤 NLP Bug Analysis | TF-IDF keyword extraction, KMeans clustering of bug descriptions, PCA scatter plot, treemap |
| ✅ Fix Validation & Insights | Reappearing bug detection, version quality comparison, 10 actionable insights, testing priority list |

---

## 🗂️ Project Structure

```
.
├── app.py               # Main Streamlit application
├── generate_data.py     # Synthetic dataset generator
├── bugs_data.csv        # Pre-generated dataset (400 rows)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚡ Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/intelligent-app-testing-system.git
cd intelligent-app-testing-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Regenerate the dataset

```bash
python generate_data.py
```

### 4. Launch the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploying on Streamlit Cloud

### Prerequisites
- A [GitHub](https://github.com) account
- A [Streamlit Cloud](https://streamlit.io/cloud) account (free tier works)

### Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit — IATS"
   git remote add origin https://github.com/YOUR_USERNAME/intelligent-app-testing-system.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **New app**
   - Select your repository and branch
   - Set **Main file path** to `app.py`
   - Click **Deploy**

3. **Done!** Streamlit Cloud automatically reads `requirements.txt` and installs dependencies. The app is live within ~2 minutes.

> **Note:** `bugs_data.csv` must be committed to your repository, or users can upload it via the sidebar file uploader.

---

## 🧠 Technical Details

### Dataset (bugs_data.csv)
- **400 rows** across 6 app versions (`1.0.0` → `2.2.0`) and 10 modules
- Columns: `Bug ID`, `App Version`, `Module`, `Bug Description`, `Severity`, `Status`, `Occurrences`, `Time to Fix (days)`, `Release Date`, `Report Date`
- Realistic severity/reopen biases per module (e.g., Payment Gateway skews High severity)

### Risk Scoring Formula
```
RiskScore = 0.35 × norm(TotalOccurrences)
           + 0.35 × norm(AvgSeverityWeight)
           + 0.30 × norm(ReopenRate)
```
Scores are normalized to 0–100. Thresholds: **High ≥ 66**, **Medium 33–66**, **Low < 33**.

### ML Model
- **Algorithm:** Gradient Boosting Classifier (scikit-learn)
- **Target:** Bug Status (Open / Fixed / Reopened)
- **Features:** Module, Severity, App Version, Occurrences, Time to Fix
- **Split:** 75% train / 25% test with stratification

### NLP Pipeline
- **Vectorizer:** TF-IDF (1-gram + 2-gram, top 500 features)
- **Clustering:** KMeans (k = min(6, N//10))
- **Visualization:** PCA 2D scatter, term treemap

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Static charts |
| `plotly` | Interactive charts |
| `scikit-learn` | ML model + NLP vectorizer + KMeans |
| `wordcloud` | (Available for extension) |

All packages are pip-installable and Streamlit Cloud compatible — no conda, no OS-level dependencies.

---

## 🔧 Customization

- **Add your own data:** Upload any CSV via the sidebar that matches the column schema, or adapt the column mappings in `app.py`.
- **Swap the ML model:** Replace `GradientBoostingClassifier` with `RandomForestClassifier` in `train_model()`.
- **Adjust risk weights:** Edit the `0.35 / 0.35 / 0.30` coefficients in `compute_risk_scores()`.
- **Regenerate data:** Edit `VERSIONS`, `MODULES`, or bug templates in `generate_data.py` and rerun.

---

## 📄 License

MIT — free to use, modify, and distribute.
