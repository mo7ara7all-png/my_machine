import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")

# preview rows, dimensions, column names
print(df.head())
print(df.shape)
print(df.columns)

# Data types and missing values information
print(df.info())


#describe the data, max,mean,min,std..
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Check duplicates
print(df.duplicated().sum())

# Data types
print(df.dtypes)

# Display the distribution of values for numerical features
df.hist(figsize=(20,15))
plt.show()

# Correlation matrix
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(18,12))
sns.heatmap(corr_matrix, cmap="coolwarm")
plt.show()

# Shares distribution
sns.histplot(df[' shares'], bins=50)
plt.show()

# Boxplot for shares
sns.boxplot(x=df[" shares"])
plt.show()

# Relationship between content length and shares
sns.scatterplot(x=' n_tokens_content', y=' shares', data=df)
plt.show()

# Shares distribution
sns.kdeplot(df[' shares'], fill=True)
plt.show()


sns.pairplot(df[[' shares',' n_tokens_content',' num_hrefs']])
plt.show()

plt.figure(figsize=(15,12))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()
