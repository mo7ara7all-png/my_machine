import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, Lasso


# =====================================================
# READ THE DATA
# =====================================================

df = pd.read_csv("data//OnlineNewsPopularity.csv")
df.columns = df.columns.str.strip()
df.drop(columns=['url'], inplace=True)
median_shares = df['shares'].median()
df['popular'] = (df['shares'] > median_shares).astype(int)
X = df.drop(columns=['shares', 'popular'])
y = df['popular']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

logistic_model = LogisticRegression(
    penalty='l2',
    solver='liblinear',
    max_iter=3000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

# =====================================================
# 1. FEATURE IMPORTANCE RANKING FOR TREE-BASED MODEL
# =====================================================

print("\n================ TREE-BASED FEATURE IMPORTANCE ================")

importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nTOP 15 IMPORTANT FEATURES:\n")
print(importance_df.head(15))


# =====================================================
# PLOT FEATURE IMPORTANCE
# =====================================================

plt.figure(figsize=(12, 7))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df.head(15)
)

plt.title("Top 15 Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


## =====================================================
# 2. SHAP SUMMARY PLOT FOR TREE-BASED MODEL
# =====================================================

print("\n================ SHAP SUMMARY PLOT ================")

X_test_sample = X_test.sample(
    n=200,
    random_state=42
)

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_test_sample)

print("SHAP values type:", type(shap_values))

if isinstance(shap_values, list):
    shap_values_class_1 = shap_values[1]
else:
    shap_values_class_1 = shap_values

shap.summary_plot(
    shap_values_class_1,
    X_test_sample,
    show=False
)

plt.title("SHAP Summary Plot - Random Forest")
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.show()


# =====================================================
# 3. SHAP DEPENDENCE PLOT
# =====================================================

print("\n================ SHAP DEPENDENCE PLOT ================")

top_feature = importance_df.iloc[0]['Feature']

print("Most important feature used in dependence plot:", top_feature)

shap.dependence_plot(
    top_feature,
    shap_values_class_1,
    X_test_sample,
    show=False
)

plt.title("SHAP Dependence Plot")
plt.tight_layout()
plt.savefig("shap_dependence_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# =====================================================
# 4. COEFFICIENT ANALYSIS FOR LINEAR MODEL
# =====================================================

print("\n================ LOGISTIC REGRESSION COEFFICIENT ANALYSIS ================")

coefficients = logistic_model.coef_[0]

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Absolute_Coefficient': np.abs(coefficients)
}).sort_values(by='Absolute_Coefficient', ascending=False)

print("\nTOP 15 LOGISTIC REGRESSION COEFFICIENTS:\n")
print(coef_df.head(15))


# =====================================================
# PLOT TOP COEFFICIENTS
# =====================================================

plt.figure(figsize=(12, 7))

sns.barplot(
    x='Coefficient',
    y='Feature',
    data=coef_df.head(15)
)

plt.title("Top 15 Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# =====================================================
# 5. REGULARIZATION EFFECTS
# =====================================================

print("\n================ REGULARIZATION EFFECTS ================")

logistic_l1 = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    max_iter=3000,
    random_state=42
)

logistic_l2 = LogisticRegression(
    penalty='l2',
    solver='liblinear',
    max_iter=3000,
    random_state=42
)

logistic_l1.fit(X_train_scaled, y_train)

logistic_l2.fit(X_train_scaled, y_train)

coef_l1 = logistic_l1.coef_[0]

coef_l2 = logistic_l2.coef_[0]

regularization_df = pd.DataFrame({
    'Feature': X.columns,
    'L1_Coefficient': coef_l1,
    'L2_Coefficient': coef_l2,
    'Abs_L1': np.abs(coef_l1),
    'Abs_L2': np.abs(coef_l2)
})

print("\nREGULARIZATION COEFFICIENT COMPARISON:\n")
print(regularization_df.head(15))

zero_l1 = np.sum(np.isclose(coef_l1, 0))

zero_l2 = np.sum(np.isclose(coef_l2, 0))

print("\nNumber of zero coefficients using L1:", zero_l1)

print("Number of zero coefficients using L2:", zero_l2)

print("\nMean absolute coefficient using L1:", np.mean(np.abs(coef_l1)))

print("Mean absolute coefficient using L2:", np.mean(np.abs(coef_l2)))


# =====================================================
# PLOT REGULARIZATION EFFECTS
# =====================================================

regularization_summary = pd.DataFrame({
    'Regularization': ['L1', 'L2'],
    'Zero_Coefficients': [zero_l1, zero_l2],
    'Mean_Absolute_Coefficient': [
        np.mean(np.abs(coef_l1)),
        np.mean(np.abs(coef_l2))
    ]
})

plt.figure(figsize=(8, 5))

sns.barplot(
    x='Regularization',
    y='Zero_Coefficients',
    data=regularization_summary
)

plt.title("Number of Zero Coefficients: L1 vs L2")
plt.xlabel("Regularization Type")
plt.ylabel("Zero Coefficients")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

sns.barplot(
    x='Regularization',
    y='Mean_Absolute_Coefficient',
    data=regularization_summary
)

plt.title("Mean Absolute Coefficient: L1 vs L2")
plt.xlabel("Regularization Type")
plt.ylabel("Mean Absolute Coefficient")
plt.tight_layout()
plt.show()


# =====================================================
# 6. FINAL EXPLAINABILITY SUMMARY
# =====================================================

print("\n================ FINAL EXPLAINABILITY SUMMARY ================")

print("\nMost important tree-based features:")
print(importance_df.head(10)['Feature'].tolist())

print("\nMost important linear-model features:")
print(coef_df.head(10)['Feature'].tolist())

print("\nInterpretation Notes:")
print("- Random Forest feature importance shows which features helped the trees make better splits.")
print("- SHAP summary plot shows both feature importance and whether feature values increase or decrease popularity prediction.")
print("- SHAP dependence plot explains how one important feature affects the prediction.")
print("- Logistic Regression coefficients show direction: positive coefficients increase popularity probability, negative coefficients decrease it.")
print("- L1 regularization can shrink some coefficients to zero, making the model simpler.")
print("- L2 regularization reduces coefficient size but usually keeps most features.")