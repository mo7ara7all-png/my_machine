# %% -------- Cross Validation & Hyperparameter Tuning --------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")
df.columns = df.columns.str.strip()


# Prepare target variable
df["log_shares"] = np.log1p(df["shares"])

y = df["log_shares"]


# Prepare features
x=df.drop(
    columns=["url", "shares", "log_shares"],
    errors="ignore"
)

#----------------------------------------------------------------------
#The test set is kept untouched until the final hold-out evaluation.
#-----------------------------------------------------------------------
# Train-test split

x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler= StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


#-------------------------------------------------------------------
# Nested CV was not used because it requires
# much longer runtime for XGBoost.
# Instead, 5-fold cross-validation with
# a separate hold-out test set was used.
#------------------------------------------------------------------------
# Cross-validation setup

cv=KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Baseline XGBoost model
baseline_model= XGBRegressor(
    random_state=42,
    objective="reg:squarederror"
)

# Baseline cross-validation
baseline_cv_scores=cross_val_score(
    baseline_model,
    x_train_scaled,
    y_train,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)

baseline_cv_rmse= -baseline_cv_scores.mean()
print("Baseline CV RMSE:",baseline_cv_rmse)


#-------------------------------------------------
# GridSearchCV tests all parameter combinations.
#--------------------------------------------------
# %% Parameter grid for GridSearchCV
param_grid ={
    "n_estimators": [100 , 200],
    "max_depth": [3 , 5],
    "learning_rate": [0.05 , 0.1],
    "subsample":[0.8 , 1.0]
}
print("GridSearchCV Parameter Grid:")
print(param_grid)


# GridSearchCV
start_time= time.time()
grid_search=GridSearchCV(
estimator=XGBRegressor(
     random_state=42,
     objective="reg:squarederror"
),

param_grid=param_grid,
cv=cv,
scoring="neg_root_mean_squared_error",
n_jobs= -1
)

grid_search.fit(x_train_scaled,y_train)
grid_runtime=time.time() -start_time

print("----- Grid SearchCV Results -----")
print("Best Parameters:",grid_search.best_params_)
print("Best CV RMSE:",-grid_search.best_score_)
print("Runtime in seconds:",grid_runtime)
#-----------------------------------------------
# GridSearchCV required more runtime
# because it tested all parameter combinations.
#------------------------------------------------


#---------------------------------------------------------
# RandomizedSearchCV samples random parameter combinations.
# It is faster than GridSearchCV for larger search spaces.
#----------------------------------------------------------
# Parameter distribution for RandomizedSearchCV
param_dist={
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate":[0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":[0.7, 0.8, 0.9, 1.0]
}
print("RandomizedSearchCV Parameter Distribution:")
print(param_dist)


# RandomizedSearchCV
start_time= time.time()

random_search= RandomizedSearchCV(
estimator=XGBRegressor(
    random_state=42,
    objective="reg:squarederror"
),
param_distributions=param_dist,
n_iter=10,
cv=cv,
scoring="neg_root_mean_squared_error",
random_state=42,
n_jobs= -1

)
random_search.fit(x_train_scaled,y_train)
random_runtime = time.time() - start_time

print("------ RandomizedSearchCV Results ------")
print("Best Parameters:",random_search.best_params_)
print("Best CV RMSE:",-random_search.best_score_)
print("Runtime in seconds:",random_runtime)
#-----------------------------------
# RandomizedSearchCV was faster
# and achieved the best RMSE.
#-----------------------------------

# Compare GridSearchCV and RandomizedSearchCV
tuning_results =pd.DataFrame({
    "Search Method":["GridSearchCV", "RandomizedSearchCV"],
    "Best CV RMSE":[
         -grid_search.best_score_,
        -random_search.best_score_
    ],
    "Runtime":[
         grid_runtime,
        random_runtime
        ]
})

print("------ Tuning Comparison ------")
print(tuning_results)

plt.figure(figsize=(8,5))

sns.barplot(
    data=tuning_results,
    x="Search Method",
    y="Best CV RMSE"
)

plt.title("GridSearchCV vs RandomizedSearchCV - CV RMSE")
plt.ylabel("Best CV RMSE")

plt.show()
plt.close()


plt.figure(figsize=(8,5))

sns.barplot(
    data=tuning_results,
    x="Search Method",
    y="Runtime"
)

plt.title("GridSearchCV vs RandomizedSearchCV - Runtime")
plt.ylabel("Runtime in Seconds")

plt.show()
plt.close()

#----------------------------------------------------------------------
# The best model is selected based on the lowest cross-validation RMSE.
#-----------------------------------------------------------------------
# Select the best tuned model

if -grid_search.best_score_ <= -random_search.best_score_:
    best_model = grid_search.best_estimator_
    best_method = "GridSearchCV"
else:
    best_model= random_search.best_estimator_
    best_method= "RandomizedSearchCV"

print("Best Search Method:", best_method)

#-----------------------------------------------------------
# The final model is evaluated on the test set only once.
# This gives an unbiased final performance estimate.
#-------------------------------------------------------------
#%% Final hold-out evaluation
best_model.fit(x_train_scaled,y_train)
final_pred=best_model.predict(x_test_scaled)

final_mae = mean_absolute_error(y_test,final_pred)
final_mse = mean_squared_error(y_test,final_pred)

final_rmse = np.sqrt(final_mse)

final_r2= r2_score(y_test,final_pred)

print("------ Final Hold-Out Evaluation ------")
print("MAE:", final_mae)
print("MSE:", final_mse)
print("RMSE:", final_rmse)
print("R2:", final_r2)


# Final comparison table
final_results = pd.DataFrame({
 "Metric": ["MAE", "MSE", "RMSE", "R2"],
 "Final Test Score":[
     final_mae,
        final_mse,
        final_rmse,
        final_r2
 ]
})

print("Final Hold-Out Results:")
print(final_results)

# Actual vs Predicted plot
plt.figure(figsize=(8,5))

sns.scatterplot(
    x=y_test,
    y=final_pred
)

plt.xlabel("Actual Log Shares")
plt.ylabel("Predicted Log Shares")
plt.title(f"Actual vs Predicted - Tuned XGBoost ({best_method})")

plt.show()
plt.close()

#-----------------------------------------------------
# Hyperparameter tuning improved model performance
# compared to the baseline XGBoost model.
#----------------------------------------------------