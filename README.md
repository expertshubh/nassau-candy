# 🍬 Nassau Candy — Factory Reallocation & Shipping Optimization System
### Supply Chain Analytics & ML Decision Support Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Models-F7931E?style=flat-square&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=flat-square&logo=plotly)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

---

## 📌 Project Overview

An intelligent ML-powered decision support system for Nassau Candy Distributor that predicts shipping lead times and recommends optimal factory-product reassignments across a **5-factory, 4-region US supply chain**.

Built on **10,194 real orders**, the system identifies factory reassignment opportunities that could reduce lead times by **up to 60 days per order** — directly impacting distribution efficiency and profitability.

> **Key Result:** Identified top reassignment targets saving up to **60 days** of lead time per order across a 5-factory supply chain.

---

## 🎯 Problem Statement

Nassau Candy operates across 5 factories serving 4 US regions. Without data-driven insight:

- Products are assigned to factories based on legacy decisions, not performance data
- High lead times on certain routes go undetected and unresolved
- No structured way to evaluate what-if reassignment scenarios
- No ranking system to prioritize which factory switches deliver the most value

This system addresses all four gaps with a live interactive dashboard.

---

## 🚀 Live Dashboard

> 🔗 **[Launch App on Streamlit Cloud](#)** ← *(https://jkxuvqrsghbsswhm6kidkz.streamlit.app/)*

**4-Tab Dashboard:**

| Tab | Description |
|---|---|
| 🏭 Factory Simulator | Compare predicted lead times across all 5 factories for any product |
| 🔄 What-If Analysis | Side-by-side comparison of current vs alternative factory |
| 🏆 Recommendations | Top 10 ranked reassignment recommendations with blended score |
| ⚠️ Risk & Impact | Profit volatility analysis and high-risk reassignment alerts |

---

## 🤖 Machine Learning Pipeline

Three models trained and evaluated on 10,194 orders:

| Model | RMSE | R² | Selected |
|---|---|---|---|
| Linear Regression | — | — | |
| Random Forest | — | — | |
| **Gradient Boosting** | **266 days** | **0.18** | ✅ Best |

**Features used:**
- Product name, Factory, Region, Ship Mode (label encoded)
- Shipping Distance (miles) — calculated via Haversine formula
- Units per order, Cost per unit

---

## 📊 Key Features

- **Blended Scoring Engine** — ranks all 60 product-factory combinations by speed improvement, profit stability, and risk reduction
- **Priority Slider** — adjust optimization from Speed-first to Profit-first in real time
- **K-Means Clustering** — identifies Congested and Slow route segments
- **Haversine Distance Calculation** — real geographic distance between factory and region
- **Download Recommendations CSV** — export top reassignment picks directly

---

## 🗂️ Project Structure

```
nassau-candy/
├── app.py                              # Streamlit dashboard (main app)
├── eda.py                              # Exploratory data analysis
├── requirements.txt                    # Python dependencies
├── Nassau_Candy_Executive_Summary.docx
├── data/
│   └── Nassau_Candy_Distributor.csv   # 10,194 orders dataset
├── models/
│   ├── best_model.pkl                 # Trained Gradient Boosting model
│   ├── encoders.pkl                   # Label encoders
│   ├── feature_cols.pkl               # Feature column list
│   └── model_results.csv             # Model comparison results
└── src/
    └── preprocess.py                  # Data cleaning & feature engineering
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/expertshubh/nassau-candy.git

# 2. Navigate into the project folder
cd nassau-candy

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn (Linear Regression, Random Forest, Gradient Boosting, K-Means) |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Distance Calc | Haversine formula |
| Model Storage | Joblib |

---

## 📋 Dataset

- **Source:** Nassau Candy Distributor internal order data
- **Records:** 10,194 orders
- **Factories:** 5 (Wicked Choccy's, The Other Factory, Secret Factory, Sugar Shack, Lot's O' Nuts)
- **Regions:** 4 US regions (Atlantic, Central, South, West)
- **Ship Modes:** Standard Class, Second Class, First Class, Same Day

---

## 💡 Key Achievements

- Identified factory reassignment opportunities reducing lead times by **up to 60 days per order**
- Built a recommendation engine scoring **all 60 product-factory combinations**
- Delivered professional stakeholder deliverables: executive summary, EDA report, live dashboard
- Full end-to-end ML pipeline from raw CSV to deployed interactive app

---

## 📄 Deliverables

- ✅ Interactive 4-tab Streamlit Dashboard
- ✅ ML Pipeline with 3 trained models and evaluation metrics
- ✅ Blended Scoring Recommendation Engine
- ✅ EDA Analysis (`eda.py`)
- ✅ Executive Summary for Stakeholders

---

## 👤 Author

**Shubham Makvana** — Data Analyst | Python | Machine Learning | Streamlit

[![GitHub](https://img.shields.io/badge/GitHub-expertshubh-black?style=flat-square&logo=github)](https://github.com/expertshubh)

---

*Supply Chain Analytics Project | Nassau Candy Distributor | 2025–2026*
