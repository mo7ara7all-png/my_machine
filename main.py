import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler



 
df = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")
df.columns = df.columns.str.strip()
df.drop(columns=['url'], inplace=True)
# print(df.head())
# print(df.shape)
# print(df.columns)

#               #-----------------------------------------------------#
#                           describe the data , null or not-null :
#               #-----------------------------------------------------#

# print(df.info())

# describe the data , min,max,mean....etc :
# print(df.describe())
#               
# ------------------------------------------"Preproccesing"----------------------------------------------#
# 
# 

#---------------------------------- check if there is a null values :-----------------------------------#
# print(df.isnull().sum())
# -------------------------------- check if there is a duplicated ROWS:----------------------------------#
# print(df.duplicated().sum())
# -------------------------------- show the duplicate Rows (if exist) :----------------------------------#
# duplicates = df[df.duplicated()]
# print(duplicates)
# ------------------------------- delete the dulicate Rows (if exist) :----------------------------------#
# df = df.drop_duplicates()

#-----------------------   We dont need encoding - because all data are numircal------------------------#




# print(df['shares'].describe())
#----------------------------------- show the data 'shares' :---------------------------------------------#

# plt.hist(df['shares'])
# plt.show
# print(df['shares'].head())
# df['shares'] = np.log1p(df['shares'])
# plt.figure(figsize=(10,5))

# plt.hist(df['shares'], bins=50)

# plt.title("Shares Distribution")
# plt.xlabel("Shares")
# plt.ylabel("Count")

# plt.show()


#-------------------------- show the relation between the feautures with 'shares' :-------------------------#
# corr = df.corr(numeric_only=True)

# shares_corr = corr['shares'].sort_values(ascending=False)

# print(shares_corr)


                
#------------------------- show the relation between the feautures with 'shares' with bar plot :----------#

# shares_corr = corr['shares'].sort_values()

# plt.figure(figsize=(10,15))

# shares_corr.plot(kind='barh')

# plt.title("Correlation with Shares")

# plt.show()


#top_corr = corr['shares'].abs().sort_values(ascending=False).head(15)

# print(top_corr)
#--------------------------------- show the data (before log transform) ---------------------------------#
# plt.figure(figsize=(12,5))

# plt.subplot(1,2,1)
# plt.hist(df['shares'], bins=50)
# plt.title("Before Log Transform")
#--------------------------------- show the data (after log transform) -----------------------------------#

log_shares = np.log1p(df['shares'])
plt.subplot(1,2,2)
plt.hist(log_shares, bins=50)
plt.title("After Log Transform")

# plt.show()
#----------------------------------- Train Data * Test Data ------------------------------------------------#
X = df.drop('shares', axis=1)

y = df['shares']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


