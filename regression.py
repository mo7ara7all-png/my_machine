# %% -----------------Regression Task-------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan



# %% Load dataset
df_clean = pd.read_csv("Data/OnlineNewsPopularity.csv")

df_clean.columns = df_clean.columns.str.strip()




# %% Prepare target variable
df_clean["log_shares"]=np.log1p(df_clean["shares"])
y=df_clean["log_shares"]

#---------------------------------------------------------------------
# use log_shares because shares contains large values and outliers.
# Log transformation helps make the target distribution more stable.
#------------------------------------------------------------------------




# %% We delete the columns that we do not want to use as inputs for the model
x=df_clean.drop(columns=["url","shares","log_shares"],errors='ignore')



# %% We split the data into training and testing
x_train,x_test,y_train,y_test=train_test_split (
    x,y,
    test_size=0.2,
    random_state=42,
    
)






#----------------------------------------------
# Feature scaling is applied because some models
# are sensitive to feature magnitudes.
#---------------------------------------------------

# %% Feature scaling
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)




# %% Function to evaluate regression models
def adjusted_r2_score(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def evaluate_regression_model(model_name, y_test, y_pred, X_test):
 mae=mean_absolute_error(y_test,y_pred)
 mse=mean_squared_error(y_test, y_pred)
 rmse=np.sqrt(mse)
 r2=r2_score(y_test,y_pred)

 n= x_test.shape[0]
 p=x_test.shape[1]
 adj_r2=adjusted_r2_score(r2,n,p)

 return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted R2": adj_r2
    }





# %% Define regression models
models = {
    "Linear Regression": LinearRegression(),
     "Ridge Regression": Ridge(alpha=1.0),
     "Lasso Regression": Lasso(alpha=0.001, max_iter=10000),
     "Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Support Vector Regressor": SVR(kernel='linear'),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost Regressor": XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
}




# %% Train, predict, and evaluate all regression models
results = []
predictions = {}

for model_name, model in models.items():
    print("Training:", model_name)

    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)

    predictions[model_name] = y_pred

    result = evaluate_regression_model(
        model_name,
        y_test,
        y_pred,
        x_test
    )

    results.append(result)




# %% Let's convert the results into a table to compare the models
results_df=pd.DataFrame(results)
results_df=results_df.sort_values(by="RMSE")
print(results_df)





# %% Plotting a comparison between models based on RMSE
plt.figure(figsize=(12,6))

sns.barplot(
    data=results_df,
    x="Model",
    y="RMSE",
)
plt.xticks(rotation=90)
plt.title("Regression Models Comparison Based on RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")

plt.show()
plt.close()





#%% We choose the best model according to the lowest RMSE
best_model_name = results_df.iloc[0]["Model"]

best_y_pred = predictions[best_model_name]

print("Best Regression Model:", best_model_name)





# %% Compares the actual values with the expected values
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=best_y_pred)


plt.xlabel("Actual Log Shares")
plt.ylabel("Predicted Log Shares")
plt.title(f"Actual vs Predicted - {best_model_name}")

plt.show()
plt.close()







#----------------------------------------------------------
# Residual analysis helps evaluate prediction errors.
#----------------------------------------------------------

#%% We calculate residuals, which are the difference between the actual value and the expected value
residuals = y_test - best_y_pred

plt.figure(figsize=(8,6))

sns.scatterplot(x=best_y_pred, y=residuals)

plt.axhline(0, color='red', linestyle='--')

plt.xlabel("Predicted Log Shares")
plt.ylabel("Residuals")
plt.title(f"Residual Plot - {best_model_name}")

plt.show()
plt.close()



#%% The chart shows the distribution of errors
plt.figure(figsize=(8,6))

sns.histplot(residuals, kde=True, bins=40)

plt.title(f"Residuals Distribution - {best_model_name}")
plt.xlabel("Residuals")

plt.show()
plt.close()






#%% This plot checks if the errors are close to a normal distribution
stats.probplot(residuals, dist="norm", plot=plt)

plt.title(f"Q-Q Plot - {best_model_name}")

plt.show()
plt.close()



#%% The Breusch-Pagan test checks whether the variance of the errors is constant or not
x_test_const = sm.add_constant(x_test_scaled)

bp_test = het_breuschpagan(residuals, x_test_const)

bp_labels = [
    "Lagrange Multiplier Statistic",
    "p-value",
    "F-Statistic",
    "F-Test p-value"
]

bp_results = dict(zip(bp_labels, bp_test))

print("Breusch-Pagan Test Results:")
print(bp_results)



#%% If the p-value is less than 0.05, this indicates the presence of heteroscedasticity
if bp_results["p-value"] < 0.05:
    print("Heteroscedasticity detected.")
else:
    print("No strong heteroscedasticity.")

