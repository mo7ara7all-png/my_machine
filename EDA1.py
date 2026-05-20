# %% Import libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler







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













