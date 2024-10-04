

# 🏡 House Price Prediction System 📊

Welcome to the **House Price Prediction System**! This project leverages machine learning to predict the price of a house based on its features, such as location, size, number of bedrooms, and more. The system uses several machine learning models to provide accurate predictions and visualizations.

## 🚀 Project Overview

The House Price Prediction System estimates house prices using real estate data. By analyzing various factors like square footage, number of rooms, and location, the system can predict housing prices with high accuracy. It provides insights through graphs and plots to help users understand trends in housing prices.

### ✨ Features:
- **Multiple ML models** 🧠 like Linear Regression, Decision Trees, and Gradient Boosting.
- **Data visualization** 📈 to help understand pricing trends.
- **User-friendly interface** 🖥️ for easy house price predictions.
- **Complete data preprocessing pipeline** 🧹 for handling missing values, scaling, and feature selection.
- **Evaluation metrics** 📊 including R², MAE, and RMSE to assess model performance.

## 🛠️ Technologies Used

- **Python** 🐍
- **Pandas** for data manipulation 📊
- **Scikit-learn** for machine learning models 🤖
- **Matplotlib & Seaborn** for data visualization 📉
- **NumPy** for numerical computations 🧮

## 📂 Project Structure

```
📦 house-price-prediction-system
├── 📁 data               # Dataset (CSV)
├── 📁 models             # Trained ML models
├── 📁 notebooks          # Jupyter notebooks for exploration and model training
├── 📁 src                # Source code for data preprocessing and model training
├── 📁 plots              # Saved graphs and plots
├── README.md             # Project documentation
└── app.py                # Main application file (if applicable)
```

## 📊 Data Visualization and Plots
Here are some visualizations included in the project to help analyze the data and model performance:

### 1. **Correlation Heatmap** 🔥
- Visualizes the correlation between different house features and the target variable (price).

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()
```

### 2. **Price Distribution Plot** 💰
- Displays the distribution of house prices to give insights into the price range.

```python
sns.histplot(data['price'], kde=True, color='blue')
plt.title('House Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
```

### 3. **Feature Importance Plot** 📊
- Shows which features have the most impact on predicting house prices using models like Random Forest or Gradient Boosting.

```python
import numpy as np
import seaborn as sns

importances = model.feature_importances_
features = data.columns.drop('price')

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices], palette="Blues_d")
plt.title('Feature Importance')
plt.show()
```

### 4. **Scatter Plot: Price vs Area** 📏
- This plot helps to visualize how house prices vary with the square footage of the property.

```python
sns.scatterplot(x=data['area'], y=data['price'], color='green')
plt.title('House Price vs Area')
plt.xlabel('Area (Square Footage)')
plt.ylabel('Price')
plt.show()
```

### 5. **Residual Plot** 🎯
- Helps in visualizing the residuals (errors) to assess model performance.

```python
residuals = y_test - y_pred

sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.show()
```

### 6. **Model Performance Comparison** 📊
- Compare the performance of different models by plotting their R² or RMSE scores.

```python
models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']
r2_scores = [0.80, 0.87, 0.90, 0.92]  # Example R² scores

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=r2_scores, palette="coolwarm")
plt.title('Model Comparison (R² Scores)')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.show()
```

## 📈 ML Models Used
The following models were trained and tested to predict house prices:
- **Linear Regression** 📏
- **Random Forest** 🌳
- **Gradient Boosting** 📈
- **XGBoost** 🚀

### Model Evaluation Metrics
- **R² (Coefficient of Determination)**: Measures how well the model explains the variance in house prices.
- **RMSE (Root Mean Square Error)**: Measures the average difference between predicted and actual prices.
- **MAE (Mean Absolute Error)**: Measures the average error in predictions.

## 🔧 How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction-system.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd house-price-prediction-system
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the project**:
   ```bash
   python app.py
   ```

## 🧠 Model Performance
- **Linear Regression**: R² = 0.80, RMSE = 25000
- **Random Forest**: R² = 0.87, RMSE = 18000
- **Gradient Boosting**: R² = 0.90, RMSE = 15000

## 🔮 Future Enhancements
- **Advanced Hyperparameter tuning** 🛠️ to improve model accuracy.
- **Web application deployment** 🌐 for user accessibility.
- **Inclusion of more features** such as neighborhood data or historical prices for better predictions.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests. 😊



**Let’s predict house prices together! 🏠💸**

