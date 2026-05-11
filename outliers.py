#------------------------------This is the way to detect the outliers-------------------------------------#
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


df = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")
df.columns = df.columns.str.strip()
df.drop(columns=['url'], inplace=True)

#1

Q1 = df['shares'].quantile(0.25)
Q3 = df['shares'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#2

outliers = df[
    (df['shares'] < lower_bound) |
    (df['shares'] > upper_bound)
]

print(outliers.shape)

#(4541, 60) before delete outliers ..

#show in BoxPlots :

sns.boxplot(x=df['shares'])
plt.show()

