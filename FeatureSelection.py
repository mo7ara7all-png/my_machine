import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

# Read data , delete the spaces , drop url colomn , make target : shares ,
#  train-test-split , feature scaling 

df = pd.read_csv("Data//OnlineNewsPopularity.csv")
df.columns = df.columns.str.strip()
df.drop(columns=['url'], inplace=True)
X = df.drop("shares", axis=1)
y = df["shares"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#1.CORRELATION-BASED FILTERING (does not need model to tarin)

print("\n================ CORRELATION =================")

correlation = df.corr(numeric_only=True)

target_corr = correlation["shares"].sort_values(ascending=False)

print(target_corr)

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    cmap="coolwarm"
)
target_corr = correlation["shares"].sort_values(ascending=False)

print(target_corr.head(15))
print(target_corr.tail(15))

plt.title("Correlation Heatmap")
plt.show()

#=====================================================
#2. RECURSIVE FEATURE ELIMINATION (RFE)
#=====================================================

print("\n================ RFE =================")

model = LinearRegression()    # --------> Because he gives coefficients to every feature, 
                                         # and RFE need to know coefficient ,to decide delete or keep.
rfe = RFE(
    estimator=model,
    n_features_to_select=10
)

rfe.fit(X_train_scaled, y_train)

selected_features_rfe = X.columns[rfe.support_]

print("Selected Features using RFE:\n")
print(selected_features_rfe)
print(rfe.ranking_)        # -----------------> Importance of every coloumn.

# =====================================================
# 3. TREE-BASED FEATURE IMPORTANCE
# =====================================================

print("\n================ TREE IMPORTANCE =================")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.head(15))

plt.figure(figsize=(12, 6))

plt.bar(
    importance_df["Feature"][:15],
    importance_df["Importance"][:15]
)

plt.xticks(rotation=90)

plt.title("Top 15 Important Features")

plt.show()

# =====================================================
# 4. PCA (Information presure)
# =====================================================

print("\n================ PCA =================")

pca = PCA(n_components=0.95)  

X_train_pca = pca.fit_transform(X_train_scaled)  #---------> Most important move.

X_test_pca = pca.transform(X_test_scaled)

print("Original Shape:", X_train_scaled.shape)

print("Reduced Shape:", X_train_pca.shape)

print("\nExplained Variance Ratio:\n")
print(pca.explained_variance_ratio_)

print("\nTotal Explained Variance:")
print(np.sum(pca.explained_variance_ratio_))

# =====================================================
# PCA VISUALIZATION
# =====================================================

pca_2d = PCA(n_components=2)

X_pca_2d = pca_2d.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    alpha=0.5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("PCA Projection")

plt.show()