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




#%%# %% Count duplicate rows
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













#%%------------Encoding------------------


#%% Creates a separate copy of the cleaned dataset for encoding
df_encoded = df_clean.copy()





#%% Detect categorical variables
categorical_cols = df_encoded.select_dtypes(include=['object']).columns.drop('url')

print("Categorical columns:", categorical_cols)
print("Number of categorical columns:", len(categorical_cols))





#%% Check unique values in categorical columns
for col in categorical_cols:
    print(col, ":", df_encoded[col].nunique(), "unique values")




#%% Apply encoding if categorical columns exist
if len(categorical_cols) > 0:
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    print("Encoding applied using One-Hot Encoding.")
else:
    print("No categorical variables found. Encoding is not needed.")


#%% Compare before and after encoding
print("Shape before encoding:", df_clean.shape)
print("Shape after encoding:", df_encoded.shape)



#%% Check data types after encoding
print(df_encoded.dtypes)




#%%--------------Feature Engineering------------------

#%% Creates a separate copy of the encoded dataset for feature engineering
df_fe =df_encoded.copy()



# %% Create weekend indicator feature
df_fe["is_weekend"]=(
    df_fe["weekday_is_saturday"] +
    df_fe["weekday_is_sunday"]
)



#%% Create article complexity score
df_fe["article_complexity"]=(
    df_fe["n_tokens_content"] /  (df_fe["num_hrefs"] +1)
)



# %%  Create image density feature
df_fe["image_density"]=(
    df_fe["num_imgs"] / (df_fe["n_tokens_content"] +1)
)


#%% Create video density feature
df_fe["video_density"]=(
    df_fe["num_videos"] / (df_fe["n_tokens_content"]+1)
)


# %% Create keyword density feature
df_fe["keyword_density"]=(
    df_fe["kw_avg_avg"] / (df_fe["n_tokens_content"] +1)  
)


# %% Create interaction feature between global sentiment and title sentiment
df_fe["sentiment_title_interaction"]=(
    df_fe["global_sentiment_polarity"] *
    df_fe["title_sentiment_polarity"]
)



# %% Create interaction feature between content length and number of images
df_fe["content_image_interaction"]=(
    df_fe["n_tokens_content"] *
    df_fe["num_imgs"]
)

# %% Create log transformation feature for shares
df_fe['log_shares'] = np.log1p(df_fe['shares'])


#%% Display the new engineered features
print(df_fe[['is_weekend',
             'article_complexity',
             'image_density',
             'video_density',
             'keyword_density',
             'sentiment_title_interaction',
             'content_image_interaction',
             'log_shares']].head())


# %% Compare dataset shape before and after feature engineering
print("Shape before feature engineering:", df_encoded.shape)
print("Shape after feature engineering:", df_fe.shape)










#%%--------------Feature Selection------------------
# --------------Correlation-based filtering -------------------


#%% Creates a copy of the feature engineered dataset for feature selection
df_fs = df_fe.copy()



# %% Calculate correlation of all numeric features with shares
shares_corr =df_fs.corr(numeric_only=True)['shares'].sort_values(ascending=False)



# %% Display top positively correlated features
print("Top positive correlations with shares")
print(shares_corr.head(10))



# %% Display top negatively correlated features
print("Top negative correlations with shares:")
print(shares_corr.tail(10))



# %% Visualize correlation of features with shares
plt.figure(figsize=(10,8))

shares_corr.drop('shares').sort_values().plot(kind='barh')

plt.title("Feature Correlation with Shares")
plt.xlabel("Correlation")
plt.ylabel("Features")

plt.show()
plt.close()






#%%--------------Recursive Feature Elimination (RFE)------------------
# Prepare features and target for RFE
X = df_fs.drop(columns=['shares', 'log_shares', 'url'], errors='ignore')
y = df_fs['shares']



#%% Create a Linear Regression model for RFE
rfe_model = LinearRegression()



#%% Apply RFE to select the top 10 features
rfe = RFE(estimator=rfe_model, n_features_to_select=10)

rfe.fit(X, y)



#%% Store selected features from RFE
rfe_selected_features = X.columns[rfe.support_]



#%% Display selected features
print("Selected features using RFE:")
print(rfe_selected_features)





#%%--------------Tree-based Importance Selection------------------



#%% Prepare features and target for tree-based importance
X_tree = df_fs.drop(columns=['shares', 'log_shares', 'url'], errors='ignore')
y_tree = df_fs['shares']



#%% Create Random Forest model for feature importance
tree_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

#%% Fit the Random Forest model
tree_model.fit(X_tree, y_tree)


#%% Extract feature importance values
feature_importance = pd.Series(
    tree_model.feature_importances_,
    index=X_tree.columns
).sort_values(ascending=False)




#%% Display top 10 important features
print("Top 10 important features using Random Forest:")
print(feature_importance.head(10))



#%% Visualize top 10 important features
plt.figure(figsize=(10,6))

feature_importance.head(10).sort_values().plot(kind='barh')

plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()
plt.close()





#%%--------------PCA / Dimensionality Reduction------------------


#%%Prepare features for PCA
X_pca = df_fs.drop(columns=['shares', 'log_shares', 'url'], errors='ignore')


#%% Standardize the features before PCA
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_pca)


#%% Apply PCA
pca = PCA()

X_pca_transformed = pca.fit_transform(X_scaled)


#%% Calculate explained variance ratio
explained_variance = pca.explained_variance_ratio_


# %% Display explained variance ratio
print("Explained Variance Ratio:")
print(explained_variance)


# %% Calculate cumulative explained variance
cumulative_variance = np.cumsum(explained_variance)



# %% Visualize cumulative explained variance
plt.figure(figsize=(10,6))

plt.plot(range(1, len(cumulative_variance)+1),
         cumulative_variance,
         marker='o')

plt.title("Dimensionality Reduction using PCA")
plt.xlabel("PCA Components")
plt.ylabel("Explained Variance Ratio")

plt.grid(True)

plt.show()
plt.close()


