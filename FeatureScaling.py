import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("OnlineNewsPopularity.csv")



X = df.drop(columns=['shares', 'popular'])
y = df['popular']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Note:
# I used Logisitic Regression because he is so sensitive to outliers.

#=============STANDARD SCALER===============#

standard_scaler = StandardScaler()

X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)

lr_standard = LogisticRegression(random_state=42)

lr_standard.fit(X_train_standard, y_train)

pred_standard = lr_standard.predict(X_test_standard)

print("=" * 50)
print("STANDARD SCALER")
print("=" * 50)

print("Accuracy:",
      accuracy_score(y_test, pred_standard))


#==============MINMAX SCALER================#

minmax_scaler = MinMaxScaler()

X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

lr_minmax = LogisticRegression(random_state=42)

lr_minmax.fit(X_train_minmax, y_train)

pred_minmax = lr_minmax.predict(X_test_minmax)

print("=" * 50)
print("MINMAX SCALER")
print("=" * 50)

print("Accuracy:",
      accuracy_score(y_test, pred_minmax))


#===============ROBUST SCALER=================#

robust_scaler = RobustScaler()

X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)

lr_robust = LogisticRegression(random_state=42)

lr_robust.fit(X_train_robust, y_train)

pred_robust = lr_robust.predict(X_test_robust)

print("=" * 50)
print("ROBUST SCALER")
print("=" * 50)

print("Accuracy:",
      accuracy_score(y_test, pred_robust))