# HousePricePredictionML_Model

# House Price Prediction — Linear Regression

> Built as part of my Data Science learning journey

## What This Project Is About

This project predicts house sale prices based on property features using Linear Regression. I used data from Kaggle to build a model that learns how features like square footage, number of bathrooms, and location characteristics affect the final sale price.

## Dataset

- Source: Kaggle — House Prices Advanced Regression
- Total rows: 1460
- Total columns: 81

## What I Did Step By Step

1. Loaded and explored the dataset from train.csv and test.csv
2. Handled missing values — dropped columns with more than 50% missing data, used median for numeric columns and "Missing" for categorical
3. Removed outliers where GrLivArea was greater than 4000 (unusually large properties)
4. Created new features — TotalSF (sum of basement, 1st floor, and 2nd floor square footage) and TotalBathrooms (full bathrooms plus half bathrooms)
5. Encoded categorical variables using get_dummies to convert text into numbers
6. Applied log transformation on the target variable (SalePrice) to handle skewed price distribution
7. Scaled features using StandardScaler to normalize the data
8. Built Linear Regression model
9. Evaluated model using RMSE on validation set

## Results

- Validation RMSE: ~20618
- Validation R² Score: ~0.9190
- This means predictions were off by about $20,618 on average, and the model explains about 92% of the variance in house prices

## Key Findings

1. Total square footage was a strong signal for house price — the model gave it high weight
2. The log transformation was important because house prices are skewed (some very expensive houses pull the average up)
3. Removing outliers (very large homes) actually helped the model perform better on typical houses

## Libraries Used

- Pandas (data loading and manipulation)
- NumPy (numerical computations)
- Scikit-learn (preprocessing, model building, evaluation)

## How To Run

1. Make sure you have the required libraries installed: `pip install pandas numpy scikit-learn`
2. Place train.csv and test.csv in the same folder as the script
3. Run the script: `python house_price_prediction.py`
4. The script will create a submission.csv file with predictions
