# %% Import libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




# %% Load dataset
df = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")
df.columns = df.columns.str.strip()



# %% preview rows, dimensions, column names
print(df.head())

#%%
print(df.shape)

#%%
print(df.columns)



# %% Data types and missing values information
print(df.info())



# %% describe the data, max,mean,min,std..
print(df.describe())



# %% Check missing values
print(df.isnull().sum())



# %% Check duplicates
print(df.duplicated().sum())



# %% Data types
print(df.dtypes)



# %% Displays the memory usage of each column
print(df.memory_usage(deep=True))



# %% Display the distribution of values for numerical features
df.hist(figsize=(20,15))
plt.show()
plt.close()


#%% Correlation matrix
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(18,12))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.show()
plt.close()


#%% Shares distribution
sns.histplot(df['shares'], bins=50)
plt.show()
plt.close()



# %% Boxplot for shares
sns.boxplot(x=df["shares"])
plt.show()
plt.close()


# %% Relationship between content length and shares
sns.scatterplot(x='n_tokens_content', y='shares', data=df)
plt.show()
plt.close()


# %% Shares distribution
sns.kdeplot(df['shares'], fill=True)
plt.show()
plt.close()


# %% Log transformation of shares
df['log_shares'] = np.log1p(df['shares'])

plt.figure(figsize=(10,6))
sns.histplot(df['log_shares'], bins=50, kde=True)
plt.title("Log Transformed Shares Distribution")
plt.xlabel("Log Shares")
plt.ylabel("Frequency")
plt.show()
plt.close()

sns.pairplot(df[['shares',
                 'n_tokens_content',
                 'num_hrefs',
                 'num_imgs',
                 'global_sentiment_polarity']])


plt.figure(figsize=(15,12))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()
plt.close()


# %% Finds all topic/category columns

topic_cols=[col for col in df.columns if 'data_channel_is' in col]




# %% Calculates average shares for each topic
topic_avg = {}

for col in topic_cols:
    topic_avg[col] = df[df[col] == 1]['shares'].mean()

topic_avg = pd.Series(topic_avg).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=topic_avg.index, y=topic_avg.values)
plt.xticks(rotation=90)
plt.title("Average Shares by Topic")
plt.xlabel("Topic")
plt.ylabel("Average Shares")
plt.show()
plt.close()


#%% Calculates average shares for each publishing day
day_avg = {}

day_cols = [col for col in df.columns if 'weekday_is' in col] # Finds all weekday columns

for col in day_cols:
    day_avg[col] = df[df[col] == 1]['shares'].mean()

day_avg = pd.Series(day_avg).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=day_avg.index, y=day_avg.values)
plt.xticks(rotation=90)
plt.title("Average Shares by Publishing Day")
plt.xlabel("Publishing Day")
plt.ylabel("Average Shares")
plt.show()
plt.close()


# %% Shows the relationship between sentiment polarity and shares
sns.scatterplot(x='global_sentiment_polarity', y='shares', data=df)
plt.title("Global Sentiment Polarity vs Shares")
plt.xlabel("Global Sentiment Polarity")
plt.ylabel("Shares")
plt.show()
plt.close()



# %% Make a copy of the original dataset before cleaning
df_clean = df.copy()
print(df_clean.head())



#%% Check missing values count
missing_count = df_clean.isnull().sum()
print(missing_count)



# %% Check missing values percentage
missing_percentage = (df_clean.isnull().sum() / len(df_clean)) * 100

missing_summary = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percentage': missing_percentage
})

print("Missing Values Summary:")
print(missing_summary[missing_summary['Missing Count'] > 0])




# %% Visualize missing values
plt.figure(figsize=(12,6))
sns.heatmap(df_clean.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()
plt.close()



# %%# Check if there are any missing values
total_missing = df_clean.isnull().sum().sum()
print("Total missing values:", total_missing)




#%% Count duplicate rows
duplicates_count = df_clean.duplicated().sum()

print("Duplicate rows:", duplicates_count)



# %%# Remove duplicates if found
if duplicates_count > 0:
    df_clean = df_clean.drop_duplicates()
    print("Duplicates removed.")
else:
    print("No duplicate rows found.")




# %% Verify shape after cleaning
print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)




#%% Target column
target_col = "shares"


#%% Calculate IQR bounds
Q1 = df_clean[target_col].quantile(0.25)
Q3 = df_clean[target_col].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)


#%% Detect outliers
outliers = df_clean[
    (df_clean[target_col] < lower_bound) |
    (df_clean[target_col] > upper_bound)
]

print("Number of outliers:", outliers.shape[0])
print("Percentage of outliers:", round(outliers.shape[0] / len(df_clean) * 100, 2), "%")


# %% Visualize outliers before treatment

plt.figure(figsize=(10, 4))
sns.boxplot(x=df_clean[target_col])
plt.title("Shares Outliers Before Treatment")
plt.xlabel("Shares")
plt.show()
plt.close()


# %% Strategy 1: Remove outliers

df_removed = df_clean[
    (df_clean[target_col] >= lower_bound) &
    (df_clean[target_col] <= upper_bound)
].copy()

print("Original shape:", df_clean.shape)
print("After removing outliers:", df_removed.shape)


# %% Strategy 2: Winsorization

df_winsorized = df_clean.copy()

df_winsorized[target_col] = df_winsorized[target_col].clip(
    lower=lower_bound,
    upper=upper_bound
)

print("Winsorization completed.")


# %% Compare distributions after treatment

plt.figure(figsize=(10, 4))
sns.boxplot(x=df_removed[target_col])
plt.title("Shares After Removing Outliers")
plt.xlabel("Shares")
plt.show()
plt.close()

plt.figure(figsize=(10, 4))
sns.boxplot(x=df_winsorized[target_col])
plt.title("Shares After Winsorization")
plt.xlabel("Shares")
plt.show()
plt.close()


# %% Function to evaluate influence on model performance

def evaluate_outlier_strategy(data, strategy_name):
    data = data.copy()
    
    data["log_shares"] = np.log1p(data["shares"])
    
    y = data["log_shares"]
    X = data.drop(columns=["url", "shares", "log_shares"], errors="ignore")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "Strategy": strategy_name,
        "Rows": data.shape[0],
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# %% Compare performance before and after outlier treatment

outlier_results = []

outlier_results.append(
    evaluate_outlier_strategy(df_clean, "Before Treatment")
)

outlier_results.append(
    evaluate_outlier_strategy(df_removed, "After Removal")
)

outlier_results.append(
    evaluate_outlier_strategy(df_winsorized, "After Winsorization")
)

outlier_results_df = pd.DataFrame(outlier_results)

print(outlier_results_df)


# %% Visualize performance comparison

plt.figure(figsize=(8, 5))
sns.barplot(data=outlier_results_df, x="Strategy", y="RMSE")
plt.title("Effect of Outlier Treatment on RMSE")
plt.xlabel("Outlier Treatment Strategy")
plt.ylabel("RMSE")
plt.xticks(rotation=20)
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=outlier_results_df, x="Strategy", y="R2")
plt.title("Effect of Outlier Treatment on R²")
plt.xlabel("Outlier Treatment Strategy")
plt.ylabel("R²")
plt.xticks(rotation=20)
plt.show()
plt.close()


























# %% Regression Task

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan



# %% Prepare target variable
df_clean["log_shares"]=np.log1p(df_clean["shares"])
y=df_clean["log_shares"]




# %% We delete the columns that we do not want to use as inputs for the model
x=df_clean.drop(columns=["url","shares","log_shares"],errors='ignore')



# %% We split the data into training and testing
x_train,x_test,y_train,y_test=train_test_split (
    x,y,
    test_size=0.2,
    random_state=42,
    
)


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
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}




# %% Train, predict, and evaluate all regression models
results =[]
predictions ={}

for model_name, model in models.items():
    print("Training:", model_name)
    model.fit(x_train_scaled,y_train)
y_pred =model.predict(x_test_scaled)
predictions[model_name]=y_pred

result=evaluate_regression_model(
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















# %% Regression Task

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan



# %% Prepare target variable
df_clean["log_shares"]=np.log1p(df_clean["shares"])
y=df_clean["log_shares"]




# %% We delete the columns that we do not want to use as inputs for the model
x=df_clean.drop(columns=["url","shares","log_shares"],errors='ignore')



# %% We split the data into training and testing
x_train,x_test,y_train,y_test=train_test_split (
    x,y,
    test_size=0.2,
    random_state=42,
    
)


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
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}




# %% Train, predict, and evaluate all regression models
results =[]
predictions ={}

for model_name, model in models.items():
    print("Training:", model_name)
    model.fit(x_train_scaled,y_train)
y_pred =model.predict(x_test_scaled)
predictions[model_name]=y_pred

result=evaluate_regression_model(
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
