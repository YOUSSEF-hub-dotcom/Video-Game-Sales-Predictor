# 🎮 Video Game Sales Predictor

## 📌 Overview
This project analyzes and predicts **video game sales** using **Machine Learning models**.
It includes **data cleaning, exploratory data analysis (EDA), visualizations, and predictive modeling**.

---

## 🛠️ Data Preparation
- Filled missing values in `Year` (median) & `Publisher` (mode).
- Removed duplicates and handled categorical data with **One-Hot Encoding**.
- Created new feature: **Total_Sales** (sum of regional sales).

---

## 🔍 Exploratory Data Analysis (EDA)
- Top publishers & most popular game genres.
- Correlation analysis between regions.
- Visualization of sales trends across years and genres.
- Regional differences: North America dominates sales, Japan shows unique preferences.

---

## 🤖 Machine Learning Models
### 1) **Linear Regression**
- **Accuracy:** 94.4%

### 2) **Decision Tree Regressor** (max_depth=5)
- **Accuracy:** 87.6%
- **Cross-Validation Average Score:** ≈ 0.96

---

## 📊 Key Insights
- **Linear Regression** performed best with 94.4% accuracy.
- **Decision Tree** achieved 87.6%, but with stronger generalization (confirmed by CV).
- **Genre Trends:** Action, Sports, and Shooter dominate global sales.
- **Regional Insights:** NA market leads global sales, Japan differs in preferences.

---

## 🚀 Next Steps
- Try **Random Forest & XGBoost** for improved performance.
- Apply **Ridge/Lasso Regularization** on Linear Regression.
- Feature Importance analysis (e.g., SHAP values).

---

## 💻 Tech Stack
- Python 🐍
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

👨‍💻 Author: **Youssef Mahmoud**
linkedin(www.linkedin.com/in/engineer-youssef-mahmoud-63b243361)
