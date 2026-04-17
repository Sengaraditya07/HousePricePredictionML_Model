import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


# LOAD
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# IDs
test_ids = df_test["Id"]

# FILTER
missing_percent = df_train.isnull().mean() * 100
good_cols = missing_percent[missing_percent < 50].index
df_train = df_train[good_cols]
df_test = df_test[good_cols.drop("SalePrice", errors='ignore')]

# MISSING
num_cols = df_train.select_dtypes(include=['int64','float64']).columns
cat_cols = df_train.select_dtypes(include=['object']).columns
for col in num_cols:
    if col != "SalePrice":
        df_train[col] = df_train[col].fillna(df_train[col].median())
        df_test[col] = df_test[col].fillna(df_train[col].median())

for col in cat_cols:
    df_train[col] = df_train[col].fillna("Missing")
    df_test[col] = df_test[col].fillna("Missing")

# ENCODE
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# FEATURES
df_train["TotalSF"] = df_train["TotalBsmtSF"] + df_train["1stFlrSF"] + df_train["2ndFlrSF"]
df_test["TotalSF"] = df_test["TotalBsmtSF"] + df_test["1stFlrSF"] + df_test["2ndFlrSF"]

df_train["TotalBathrooms"] = df_train["FullBath"] + (0.5 * df_train["HalfBath"])
df_test["TotalBathrooms"] = df_test["FullBath"] + (0.5 * df_test["HalfBath"])

# REALIGN
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# OUTLIERS
df_train = df_train[df_train["GrLivArea"] < 4000]

# SPLIT
X = df_train.drop("SalePrice", axis=1)
y = df_train["SalePrice"]

df_test = df_test.drop("SalePrice", axis=1, errors='ignore')

# SCALE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_test_scaled = scaler.transform(df_test)

# VALIDATE
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

y_train_log = np.log(y_train)
y_val_log = np.log(y_val)

model = LinearRegression()
model.fit(X_train, y_train_log)
y_pred = model.predict(X_val)
y_pred = np.exp(y_pred)
y_val_actual = np.exp(y_val_log)

rmse = np.sqrt(mean_squared_error(y_val_actual, y_pred))
print("Validation RMSE:", rmse)
r2s = r2_score(y_val_actual, y_pred)
print("Validation R²:", r2s)

# TRAIN
y_full = np.log(y)

model.fit(X_scaled, y_full)

# PREDICT
predictions = model.predict(df_test_scaled)
predictions = np.exp(predictions)

# SUBMIT
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": predictions
})

submission.to_csv("submission.csv", index=False)

print("Submission file created")