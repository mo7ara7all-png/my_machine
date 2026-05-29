# %% --------------------Robustness & Stress Testing ----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score





# %% Load dataset
df_clean = pd.read_csv("Data/OnlineNewsPopularity.csv")
df_clean.columns = df_clean.columns.str.strip()




# %% Prepare data for robustness testing
df_robust =df_clean.copy()


# %% Convert shares to log_shares to reduce skewness
df_robust["log_shares"]= np.log1p(df_robust["shares"])
y=df_robust["log_shares"]

x=df_robust.drop(
    columns=["url","shares","log_shares"],
    errors="ignore"
    )



# %% Split data into training and testing sets

x_train,x_test,y_train,y_test=train_test_split(
x,
y,
test_size=0.2,
random_state=42
)



# %% Scale features before training the model

scaler= StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)



# %% Function to train and evaluate the model
def evaluate_model(x_train_data,y_train_data,x_test_data,y_test_data):


    #--------------------------------------------
    # We use Ridge Regression because it is stable
    # and works well for testing model robustness.
    #---------------------------------------------

    model=Ridge(alpha=1.0)
    model.fit(x_train_data,y_train_data)

    y_pred=model.predict(x_test_data)

    rmse=np.sqrt(
        mean_squared_error(y_test_data,y_pred)
        )
    
    r2=r2_score(y_test_data,y_pred)
    return rmse,r2


# %% Train the baseline model without any changes

baseline_rmse, baseline_r2 =evaluate_model(
    x_train_scaled,
    y_train,
    x_test_scaled,
    y_test
)

print("Baseline RMSE:" , baseline_rmse)
print("Baseline R2:", baseline_r2)



# %% Add Gaussian noise to test model stability

noise_levels={
    "Low Noise": 0.05,
    "Medium Noise": 0.10,
    "High Noise": 0.20
}

noise_results=[]

noise_results.append({
    "Scenario": "Baseline",
    "RMSE":baseline_rmse,
    "R2":baseline_r2
    
})


for scenario, noise_factor in noise_levels.items():
      noise=noise_factor * np.random.normal(
          loc=0,
          scale=1,
          size=x_train_scaled.shape
      )
      x_train_noisy=x_train_scaled + noise
      rmse,r2=evaluate_model(
        x_train_noisy,
        y_train,
        x_test_scaled,
        y_test
      )

      noise_results.append({
           "Scenario": scenario,
           "RMSE": rmse,
           "R2": r2
       })
      
      noise_results_df=pd.DataFrame(noise_results)

      print("Noise Injection Results:")
      print(noise_results_df)








# %% Compare performance after adding noise

plt.figure(figsize=(10,7))
sns.barplot(
    data=noise_results_df,
    x="Scenario",
    y="RMSE"
    )
plt.title("Noise Injection - RMSE")

plt.show()
plt.close()


plt.figure(figsize=(10,7))

sns.barplot(
    data=noise_results_df,
    x="Scenario",
    y="R2"
)
plt.title("Noise Injection - R²")

plt.show()
plt.close()

#-----------------------------------------------------
# Small changes in RMSE and R² show good robustness.
#-----------------------------------------------------





# %% Remove important features and evaluate performance
correlations =df_robust.drop(
    columns=["url"],
    errors="ignore"
).corr(numeric_only=True)

important_features =(
    correlations["log_shares"]
    .abs()
    .sort_values(ascending=False)
    .drop(labels=["log_shares", "shares"], errors="ignore")
    .head(5)
    .index
    .tolist()
)
print("Removed Important Features:")
print(important_features)


x_train_removed=x_train.drop(
    columns=important_features,
    errors="ignore"
)

x_test_removed=x_test.drop(
    columns=important_features,
    errors="ignore"
)

scaler_removed=StandardScaler()
x_train_removed_scaled=scaler_removed.fit_transform(x_train_removed)
x_test_removed_scaled=scaler_removed.transform(x_test_removed)
removed_rmse, removed_r2= evaluate_model(
    x_train_removed_scaled,
    y_train,
    x_test_removed_scaled,
    y_test
)

feature_removal_results_df= pd.DataFrame({
    "Scenario": ["Baseline" ," After Feature Removal"],
    "RMSE": [baseline_rmse, removed_rmse],
    "R2": [baseline_r2, removed_r2]
})

print("Feature Removal Results:")
print(feature_removal_results_df)

#---------------------------------------------------------------------
# The model performance became worse after removing important features.
#----------------------------------------------------------------------









# %% Compare performance after removing important features
plt.figure(figsize=(10,7))

sns.barplot(
    data=feature_removal_results_df,
    x="Scenario",
    y="RMSE"
)

plt.title("Feature Removal - RMSE")

plt.show()
plt.close()


plt.figure(figsize=(10,7))
sns.barplot(
    data=feature_removal_results_df,
    x="Scenario",
    y="R2"
)

plt.title("Feature Removal - R²")

plt.show()
plt.close()






# %% Train the model using smaller amounts of data

traning_sizes = {
    "10%" : 0.10,
    "30%" : 0.30,
    "50%" : 0.50,
    "100%" : 1.00
    }

reduced_results = []
for scenario, fraction in traning_sizes.items():
      
      if fraction < 1.00:
          x_subset,_,y_subset,_= train_test_split(
              x_train_scaled,
              y_train,
              train_size=fraction,
              random_state=42

          )
      else: 
          x_subset =x_train_scaled
          y_subset=y_train

      rmse,r2=evaluate_model(
          x_subset,
          y_subset,
          x_train_scaled,
          y_train
      )    

      reduced_results.append({
           "Training Size" : scenario,
           "RMSE" : rmse,
           "R2" : r2
      })

      reduced_results_df=pd.DataFrame(reduced_results)

      print("Reduced Training Data Results:")
      print(reduced_results_df)


#----------------------------------------------------------------------
# Using more training data helped the model achieve better performance.
#-------------------------------------------------------------------------





# %% Compare performance using different training sizes

plt.figure(figsize=(10,7))
sns.barplot(
    data=reduced_results_df,
    x="Training Size",
    y="RMSE"

)

plt.title("Reduced Training Data - RMSE")

plt.show()
plt.close()


plt.figure(figsize=(10,7))
sns.barplot(
    data=reduced_results_df,
    x="Training Size",
    y="R2" 
)
plt.title("Reduced Training Data - R²")

plt.show()
plt.close()

#--------------------------------------------------------------
# The model performed better when more training data was used.
#---------------------------------------------------------------





# %% Add artificial outliers to test model sensitivity

outlier_levels ={
    "Low Outliers": 0.01,
    "Medium Outliers": 0.03,
    "High Outliers": 0.05
}

outlier_results =[]
outlier_results.append({
    "Scenario" : "Baseline",
    "RMSE" : baseline_rmse,
    "R2" : baseline_r2
})

for scenario, fraction in outlier_levels.items():
    y_train_outlier =y_train.copy()
    n_outliers = int(len(y_train_outlier) * fraction)
    np.random.seed(42)

    outlier_indices =np.random.choice(
        y_train_outlier.index,
        size=n_outliers,
        replace=False
    )

    y_train_outlier.loc[outlier_indices] =(
         y_train_outlier.loc[outlier_indices] + 3
    )
  
    rmse,r2=evaluate_model(
        x_train_scaled,
        y_train_outlier,
        x_test_scaled,
        y_test
    )

    outlier_results.append({
        "Scenario" : scenario,
        "RMSE" : rmse,
        "R2" : r2 
    })

    outlier_results_df=pd.DataFrame(outlier_results)
    print("Outlier Sensitivity Results:")
    print(outlier_results_df)

#------------------------------------------------------
# The model became less accurate
# after adding artificial outliers.
#------------------------------------------------------






# %% Compare performance after adding outliers

plt.figure(figsize=(10,7))
sns.barplot(
    data=outlier_results_df,
    x="Scenario",
    y="RMSE"

)
plt.title("Outlier Sensitivity - RMSE")

plt.show()
plt.close()


plt.figure(figsize=(10,7))
sns.barplot(
    data=outlier_results_df,
    x="Scenario",
    y="R2"
)
plt.title("Outlier Sensitivity - R2")

plt.show()
plt.close()

#------------------------------------------------------------
# Adding outliers increased prediction error
# and reduced model performance.
#------------------------------------------------------------





#-------------------------------------------------------
# Robustness testing was performed using:
# 1. Noise Injection
# 2. Feature Removal
# 3. Reduced Training Data
# 4. Outlier Sensitivity
#--------------------------------------------------------

