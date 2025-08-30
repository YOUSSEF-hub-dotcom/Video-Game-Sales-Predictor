from  sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\youss\Downloads\vgsales.csv")
pd.set_option('display.width',None)
print(df.head(51))

print("----------------------------------")
print("=========>>> Basic Function:")
print("number of rows and columns:")
print(df.shape)
print("The name of columns:")
print(df.columns)
print("Data types in Dataset:")
print(df.dtypes)
print("Information About Data:")
print(df.info())
print("Statistical Operations:")
print(df.describe().round())
print("number of frequency rows:")
print(df.duplicated().sum())
print("----------------------------------")
print("=========>>> Cleaning Data:")
missing_percentage = df.isnull().mean() * 100
print("The percentage of missing values in every column:\n",missing_percentage)
print("number of missing values before Cleaning:")
print(df.isnull().sum())
print("The missing Values in Year columns is 1.6 % so we use fillna:")
df['Year'] = df['Year'].fillna(df['Year'].median()).astype(int)
print("The missing Values in Publisher columns is 0.35 % so we use mode:")
df['Publisher'] = df['Publisher'].fillna(df['Publisher'].mode()[0])
print("The Year,and Publisher columns contain miss values?")
print(df[['Year','Publisher']].isnull().sum())
print("number of missing values after Cleaning:")
print(df.isnull().sum())
sns.heatmap(df.isnull())
plt.title("Missing Values in Dataset")
plt.show()
print("----------------------------------")
print("=========>>> Exploration Data Analysis:")
#Index(['Rank', 'Name', 'Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales','EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
#-------------------------------
print("Who is The publisher the Super Mario Bros. game:")
print(df[df['Name'] == 'Super Mario Bros.'] ['Publisher'])
print("How many unique video games:")
print(df['Name'].nunique())
print("What are  unique video games:")
print(df['Name'].unique())
print("What are the total profit for North_America,Europe,and Japan per Year:")
print(df.groupby('Year')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].agg(['sum','mean','count']))
print("Who's The Big Publisher have a big affect on Global_Sales?")
print(df.groupby('Publisher')['Global_Sales'].agg(['sum','count','mean']))
print('what are the top 5 video games ?')
print(df['Name'].value_counts().head(5))
print('what are the last 5 video games ?')
print(df['Name'].value_counts().tail(5))
print("Who is the most publisher?")
print(df['Publisher'].max())
print("What is the name of the most publisher?")
print(df[df['Publisher'] == df['Publisher'].max()])
print("Who is the least publisher?")
print(df['Publisher'].min())
print("What is the name of the least publisher?")
print(df[df['Publisher'] == df['Publisher'].min()])

print("Select the first row:")
print(df.iloc[0,:])
print("Select the first column:")
print(df.iloc[:,0])
print("Select first three rows:")
print(df.iloc[:3,:])
print("Select last column:")
print(df.iloc[:,-1])
print("Select the last 5 columns:")
print(df.iloc[:,-5:])
print("Select the rows [3,76,18]:")
print(df.iloc[[3,76,18],:])
#-------------------------
sns.pairplot(df)
plt.show()
#------------------------
print("The max of Global_Sales:")
print(df["Global_Sales"].max())
print("The average of Global_Sales:")
print(df["Global_Sales"].mean())
print("The min of Global_Sales:")
print(df["Global_Sales"].min())
print("The highest amount of NA_Sales:")
print(df['NA_Sales'].max())
print("The last amount of NA_Sales:")
print(df['NA_Sales'].min())
print("The highest amount of EU_Sales:")
print(df['EU_Sales'].max())
print("The last amount of EU_Sales:")
print(df['EU_Sales'].min())
print("The highest amount of JP_Sales:")
print(df['JP_Sales'].max())
print("The last amount of JP_Sales:")
print(df['JP_Sales'].min())

print("What are the top 10 video games in 2006:")
print(df[df['Year'] == 2006] ['Name'].value_counts().head(10))
print("What is the total of Global_Sales according to Genre?")
print(df.groupby('Genre')['Global_Sales'].sum())
print("Global Sales Statistical:")
print(df['Global_Sales'].describe())
print("Total Sales in each region:")
regio_sales = df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum()
print(regio_sales)
print("The Percentage of Sales in every region :")
print((regio_sales / df['Global_Sales'].sum()*100).round(2))
print("Total of Sales by Genre and Region:")
genre_region_sales = df.groupby('Genre')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum()
print(genre_region_sales)
print("The most Genre sell in every Region:")
dominant_genre = genre_region_sales.idxmax()
print(dominant_genre)
print("The ratio between regional and global sales:")
df['NA_Contribution'] = df['NA_Sales'] / df['Global_Sales']
print("df['NA_Contribution']:",df['NA_Contribution'])
df['EU_Contribution'] = df['EU_Sales'] / df['Global_Sales']
print("df['EU_Contribution']:",df['EU_Contribution'])
df['JP_Contribution'] = df['JP_Sales'] / df['Global_Sales']
print("df['JP_Contribution']:",df['JP_Contribution'])
print("Average contribution of each region to Global_Sales:")
print(df[['NA_Contribution','EU_Contribution','JP_Contribution']])
"""
"""
print("----------------------------------")
print("=========>>> Visualization:")
# Bar Plot
genre_sales = df.groupby('Genre')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum()
plt.figure(figsize=(12,6))
genre_sales.plot(kind='bar',stacked=True,colormap='Set3')
plt.title("Regional Sales by Game Genre",fontsize=14,pad=15)
plt.xlabel('Game Genre',fontsize=12)
plt.ylabel("Sales(Million Unites)",fontsize=12)
plt.legend(title='Region',bbox_to_anchor = (1.05,1),loc='upper left')
plt.tight_layout()
plt.show()
# Line Plot
yearly_genre_sales = df.groupby(['Year','Genre'])['Global_Sales'].sum().unstack().fillna(0)
top_genres = df.groupby('Genre')['Global_Sales'].sum().nlargest(5).index
yearly_top_genres = yearly_genre_sales[top_genres]
plt.figure(figsize=(12,6))
for genre in top_genres:
    plt.plot(yearly_top_genres.index,yearly_genre_sales[genre],marker = 'o',label=genre)
plt.title('Global Sales Trends for 5 Genres Over Years',fontsize=14,pad=15)
plt.xlabel('Year',fontsize=12)
plt.ylabel('Global Sales (Million Units)',fontsize=12)
plt.legend(title='Game Genre')
plt.grid(True)
plt.tight_layout()
plt.show()
#Heatmap
regio_sales = df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']]
correlation_matrix = regio_sales.corr()
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',vmin=-1,vmax=1,center=0)
plt.title('Correlation Between Regional Sales',fontsize=14,pad=15)
plt.tight_layout()
plt.show()

print("=========>>> Machine Learning:")
# Visualization
sns.regplot(data=df, x='NA_Sales', y='Global_Sales')
plt.grid()
plt.show()

print("=========>>> Building Model :")

df['Total_Sales'] = df['NA_Sales'] + df['EU_Sales'] + df['JP_Sales'] + df['Other_Sales']

df = pd.get_dummies(df, columns=['Publisher', 'Platform'], drop_first=True)

X = df.drop(columns=['Global_Sales','Rank','Name','Genre'])
y = df['Global_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print("=========>>> Model Training and Prediction >> LinearRegression:")
model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print("=========>>> Building Evaluation >> LinearRegression:")
print("Mean Squared error:", mean_squared_error(y_test, y_predict))
print("Mean absolute error:", mean_absolute_error(y_test, y_predict))
print("model.score:", model.score(X, y))#94.5%
print("R2 Score:", r2_score(y_test, y_predict))
print("=============>>> Coefficients:")
print(model.coef_)
print("----------------------------------")

print("=========>>> Model Training and Prediction >> DecisionTreeRegressor:")
model_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
model_tree.fit(X_train, y_train)
y_predict_tree = model_tree.predict(X_test)

print("=========>>> Building Evaluation >> DecisionTreeRegressor:")
print("Mean Squared error:", mean_squared_error(y_test, y_predict_tree))
print("Mean absolute error:", mean_absolute_error(y_test, y_predict_tree))
print("model.score:", model_tree.score(X, y))
print("R2 Score:", r2_score(y_test, y_predict_tree))  # 87.5%
print("-----------------------------")
print('Cross Validation')
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean()) # 0.96
print('===========================================')
"""
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
-------------------------------------------------------------------------------------------
🔹 LinkedIn Description (English)

Video Game Sales Predictor 🎮
A machine learning project focused on analyzing and predicting video game sales.

Conducted data cleaning, feature engineering, and exploratory data analysis (EDA).

Built and evaluated predictive models (Linear Regression & Decision Tree).

Achieved 94.4% accuracy with Linear Regression and 87.6% accuracy with Decision Tree.

Discovered key insights: North America dominates sales, with Action, Sports, and Shooter as top genres.
------------------------------------------------------------------------------------------
🟦 LinkedIn Post

🚀 New Machine Learning Project: Video Game Sales Predictor 🎮

I’ve just completed a project where I analyzed and predicted video game sales using Machine Learning.

📊 What I did:

Cleaned and prepared the dataset.

Performed EDA & visualization to uncover sales trends.

Built and evaluated Linear Regression and Decision Tree models.

✅ Results:

Linear Regression → 94.4% Accuracy

Decision Tree → 87.6% Accuracy (CV Avg ≈ 0.96)

🔍 Insights:

North America leads global sales.

Top genres: Action, Sports, Shooter.

Publishers like Nintendo and EA dominate.

🛠️ Next Steps: experimenting with Random Forest & XGBoost.

🔗 Full project on GitHub: [Add your link here]

"""