# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection: Import essential libraries like pandas, numpy, sklearn, matplotlib, and seaborn. Load the dataset using pandas.read_csv().
2. Data Preprocessing: Address any missing values in the dataset. Select key features for training the models. Split the dataset into training and testing sets with train_test_split().
3. Linear Regression: Initialize the Linear Regression model from sklearn. Train the model on the training data using .fit(). Make predictions on the test data using .predict(). Evaluate model performance with metrics such as Mean Squared Error (MSE) and the R² score. 
4. Polynomial Regression: Use PolynomialFeatures from sklearn to create polynomial features. Fit a Linear Regression model to the transformed polynomial features. Make predictions and evaluate performance similar to the linear regression model.
5. Visualization: Plot the regression lines for both Linear and Polynomial models. Visualize residuals to assess model performance.

## Program:
```PYTHON
/*
Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: MOHAN S
RegisterNumber:  212223240094
*/
```
```
import pandas as pd
from sklearn. model_selection import train_test_split
from sklearn. linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt

df = pd.read_csv("encoded_car_data.csv")
df

X = df[['enginesize','horsepower','citympg','highwaympg']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# For Linear Regression

linear_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model',LinearRegression())
])
linear_model.fit(X_train,y_train)
y_pred_linear = linear_model.predict(X_test)

# For Polynomial Regression

poly_model = Pipeline([
    ('Poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('model',LinearRegression())
])
poly_model.fit(X_train,y_train)
y_pred_poly = poly_model.predict(X_test)

#Visualization

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_linear, label='Linear', alpha=0.6) 
plt.scatter(y_test, y_pred_poly, label='Polynomial(degree=2)', alpha=0.6) 
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction' )
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression" )
plt.legend()
plt.show()
```

## Output:
![Screenshot 2025-04-27 193935](https://github.com/user-attachments/assets/e3074ec5-c8d4-4c1e-a0a1-1dc864baf171)

### For Linear Regression:-
![Screenshot 2025-04-27 193943](https://github.com/user-attachments/assets/0c5f01f9-5fb5-4295-a885-2fa908699500)

### For Polynomial Regression:-
![Screenshot 2025-04-27 193950](https://github.com/user-attachments/assets/ff73ee05-33dd-4cb5-a9f6-aeacbbd7c470)

### Visualization:-
![Screenshot 2025-04-27 194000](https://github.com/user-attachments/assets/baea7b2e-4981-4e6f-9684-b7a619a35a5f)



## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
