import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Read The Data :
df = pd.read_csv("Data//OnlineNewsPopularity.csv")

# Clean Column Names :
# ----------------------------------------------------------------------------------------- 
df.columns = df.columns.str.strip()

# Drop The URL column (It is not That Important) :
#----------------------------------------------------------------------------------------- 
df.drop(columns=['url'], inplace=True)


# Create Classification Target :
# ----------------------------------------------------------------------------------------- 
median_shares = df['shares'].median()
df['popular'] = (df['shares'] > median_shares).astype(int)

X = df.drop(columns=['shares', 'popular'])
y = df['popular']

# Train and Test Split :
# ----------------------------------------------------------------------------------------- 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling :
# ----------------------------------------------------------------------------------------- 
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Solve The First solution required :

# Logistic Regression
# ----------------------------------------------------------------------------------------- 
print("=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)


lr_model = LogisticRegression(random_state=42)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, lr_pred))

print("Precision:",
      precision_score(y_test, lr_pred))

print("Recall:",
      recall_score(y_test, lr_pred))

print("F1 Score:",
      f1_score(y_test, lr_pred))


# Classification Report For Logistic Regression 
print("\nClassification Report:\n")

print(classification_report(
    y_test,
    lr_pred
))

# Confusion Matrix For Logistic Regression
cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(5,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# (ROC + AUC) For logistic Regression
lr_prob = lr_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    lr_prob
)

auc_score = roc_auc_score(
    y_test,
    lr_prob
)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(f"Logistic Regression ROC Curve (AUC = {auc_score:.2f})")

plt.show()

print("Logistic Regression AUC:", auc_score)


# Feature Importance For Logistic Regression
lr_importance = np.abs(lr_model.coef_[0])

lr_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lr_importance
}).sort_values(by='Importance', ascending=False)

print("\nTOP 10 IMPORTANT FEATURES:\n")
print(lr_importance_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=lr_importance_df.head(10))
plt.title("Top 10 Important Features - Logistic Regression")
plt.tight_layout()
plt.show()


# CALIBRATION CURVE + BRIER SCORE FOR LOGISTIC REGRESSION
lr_prob = lr_model.predict_proba(X_test)[:,1]

prob_true, prob_pred = calibration_curve(
    y_test,
    lr_prob,
    n_bins=10
)

plt.figure(figsize=(6,6))

plt.plot(prob_pred, prob_true)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("Predicted Probability")

plt.ylabel("True Probability")

plt.title("Logistic Regression Calibration Curve")

plt.show()

brier = brier_score_loss(y_test, lr_prob)

print("Logistic Regression Brier Score:", brier)


print("=" * 50)
print("KNN Classifier")
print("=" * 50)

# KNN Classifier :
# ----------------------------------------------------------------------------------------- 
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, knn_pred))

print("Precision:",
      precision_score(y_test, knn_pred))

print("Recall:",
      recall_score(y_test, knn_pred))

print("F1 Score:",
      f1_score(y_test, knn_pred))


# Classification Report For KNN
print("\nClassification Report:\n")

print(classification_report(
    y_test,
    knn_pred
))

# Confusion Matrix For KNN
cm = confusion_matrix(y_test, knn_pred)

plt.figure(figsize=(5,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("KNN Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# (ROC + AUC) For KNN
knn_prob = knn_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    knn_prob
)

auc_score = roc_auc_score(
    y_test,
    knn_prob
)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(f"KNN ROC Curve (AUC = {auc_score:.2f})")

plt.show()

print("KNN AUC:", auc_score)


# CALIBRATION CURVE + BRIER SCORE FOR KNN

knn_prob = knn_model.predict_proba(X_test)[:,1]

prob_true, prob_pred = calibration_curve(
    y_test,
    knn_prob,
    n_bins=10
)

plt.figure(figsize=(6,6))

plt.plot(prob_pred, prob_true)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("Predicted Probability")

plt.ylabel("True Probability")

plt.title("KNN Calibration Curve")

plt.show()

brier = brier_score_loss(y_test, knn_prob)

print("KNN Brier Score:", brier)


print("=" * 50)
print("Support Vector Machine (linear + kernel)")
print("=" * 50)


# Support Vector Machine (linear + kernel) :
# ----------------------------------------------------------------------------------------- 
# 1)
# Linear SVM.

svm_linear = SVC(
    kernel='linear',
    random_state=42
)
svm_linear.fit(X_train, y_train)

svm_pred = svm_linear.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, svm_pred))

print("Precision:",
      precision_score(y_test, svm_pred))

print("Recall:",
      recall_score(y_test, svm_pred))

print("F1 Score:",
      f1_score(y_test, svm_pred))


# Classification Report For SVM(Linear)
print("\nClassification Report:\n")

print(classification_report(
    y_test,
    svm_pred
))

# Confusion Matrix For SVM(Linear)
cm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(5,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Linear SVM Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# (ROC + AUC) For Linear SVM
svm_linear_prob = svm_linear.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    svm_linear_prob
)

auc_score = roc_auc_score(
    y_test,
    svm_linear_prob
)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(f"Linear SVM ROC Curve (AUC = {auc_score:.2f})")

plt.show()

print("Linear SVM AUC:", auc_score)

#2)
# Kernel SVM.

svm_rbf = SVC(
    kernel='rbf',
    probability=True,
    random_state=42
)
svm_rbf.fit(X_train, y_train)

svm_rbf_pred = svm_rbf.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, svm_rbf_pred))

print("Precision:",
      precision_score(y_test, svm_rbf_pred))

print("Recall:",
      recall_score(y_test, svm_rbf_pred))

print("F1 Score:",
      f1_score(y_test, svm_rbf_pred))


# Classification Report For SVM(Kernel)
print("\nClassification Report:\n")

print(classification_report(
    y_test,
    svm_rbf_pred
    
    ))


# Confusion Matrix For SVM(kernel)
cm = confusion_matrix(y_test, svm_rbf_pred)

plt.figure(figsize=(5,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Kernel SVM Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# (ROC + AUC) For Kernel SVM
svm_rbf_prob = svm_rbf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    svm_rbf_prob
)

auc_score = roc_auc_score(
    y_test,
    svm_rbf_prob
)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(f"Kernel SVM ROC Curve (AUC = {auc_score:.2f})")

plt.show()

print("Kernel SVM AUC:", auc_score)


# CALIBRATION CURVE + BRIER SCORE FOR KERNEL SVM

svm_rbf_prob = svm_rbf.predict_proba(X_test)[:,1]

prob_true, prob_pred = calibration_curve(
    y_test,
    svm_rbf_prob,
    n_bins=10
)

plt.figure(figsize=(6,6))

plt.plot(prob_pred, prob_true)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("Predicted Probability")

plt.ylabel("True Probability")

plt.title("Kernel SVM Calibration Curve")

plt.show()

brier = brier_score_loss(y_test, svm_rbf_prob)

print("Kernel SVM Brier Score:", brier)


print("=" * 50)
print("Decision Tree")
print("=" * 50)

# Decision Tree :
# ----------------------------------------------------------------------------------------- 
tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, tree_pred))

print("Precision:",
      precision_score(y_test, tree_pred))

print("Recall:",
      recall_score(y_test, tree_pred))

print("F1 Score:",
      f1_score(y_test, tree_pred))


# Classification Report For SVM(Kernal)
print("\nClassification Report:\n")

print(classification_report(
    y_test,
    tree_pred
))

# Confusion Matrix For Disicion Tree
cm = confusion_matrix(y_test, tree_pred)

plt.figure(figsize=(5,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Decision Tree Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# (ROC + AUC) For Desicion Tree
tree_prob = tree_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    tree_prob
)

auc_score = roc_auc_score(
    y_test,
    tree_prob
)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(f"Decision Tree ROC Curve (AUC = {auc_score:.2f})")

plt.show()

print("Decision Tree AUC:", auc_score)


# Feature Importance For Decision Tree
print("=" * 50)
print("Feature Importance - Decision Tree")
print("=" * 50)

tree_importance = tree_model.feature_importances_

tree_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': tree_importance
}).sort_values(by='Importance', ascending=False)

print("\nTOP 10 IMPORTANT FEATURES:\n")
print(tree_importance_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=tree_importance_df.head(10))
plt.title("Top 10 Important Features - Decision Tree")
plt.tight_layout()
plt.show()


# CALIBRATION CURVE + BRIER SCORE FOR DECISION TREE
tree_prob = tree_model.predict_proba(X_test)[:,1]

prob_true, prob_pred = calibration_curve(
    y_test,
    tree_prob,
    n_bins=10
)

plt.figure(figsize=(6,6))

plt.plot(prob_pred, prob_true)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("Predicted Probability")

plt.ylabel("True Probability")

plt.title("Decision Tree Calibration Curve")

plt.show()

brier = brier_score_loss(y_test, tree_prob)

print("Decision Tree Brier Score:", brier)


print("=" * 50)
print("Random Forest")
print("=" * 50)

# Random Forest :
# ----------------------------------------------------------------------------------------- 
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print("Precision:",
      precision_score(y_test, rf_pred))

print("Recall:",
      recall_score(y_test, rf_pred))

print("F1 Score:",
      f1_score(y_test, rf_pred))



# Classification Report For Random Forest
print("\nClassification Report:\n")

print(classification_report(
    y_test,
    rf_pred
))

# Confusion Matrix For Random Forest
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# (ROC + AUC) For Random Forest
rf_prob = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    rf_prob
)

auc_score = roc_auc_score(
    y_test,
    rf_prob
)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(f"Random Forest ROC Curve (AUC = {auc_score:.2f})")

plt.show()

print("Random Forest AUC:", auc_score)


# Feature Importance For Random Forest
print("=" * 50)
print("Feature Importance - Random Forest")
print("=" * 50)

rf_importance = rf_model.feature_importances_

rf_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_importance
}).sort_values(by='Importance', ascending=False)

print("\nTOP 10 IMPORTANT FEATURES:\n")
print(rf_importance_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=rf_importance_df.head(10))
plt.title("Top 10 Important Features - Random Forest")
plt.tight_layout()
plt.show()


# CALIBRATION CURVE + BRIER SCORE FOR RANDOM FOREST

rf_prob = rf_model.predict_proba(X_test)[:,1]

prob_true, prob_pred = calibration_curve(
    y_test,
    rf_prob,
    n_bins=10
)

plt.figure(figsize=(6,6))

plt.plot(prob_pred, prob_true)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("Predicted Probability")

plt.ylabel("True Probability")

plt.title("Random Forest Calibration Curve")

plt.show()

brier = brier_score_loss(y_test, rf_prob)

print("Random Forest Brier Score:", brier)


print("=" * 50)
print("XGBOOST CLASSIFIER")
print("=" * 50)

# XGBoost Classifier :
# -----------------------------------------------------------------------------------------

xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, xgb_pred))

print("Precision:",
      precision_score(y_test, xgb_pred))

print("Recall:",
      recall_score(y_test, xgb_pred))

print("F1 Score:",
      f1_score(y_test, xgb_pred))


# Classification Report For XGBoost
print("\nClassification Report:\n")

print(classification_report(
    y_test,
    xgb_pred
))

# Confusion Matrix For XGBoost
cm = confusion_matrix(y_test, xgb_pred)

plt.figure(figsize=(5,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("XGBoost Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# (ROC + AUC) For XGBoost
xgb_prob = xgb_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    xgb_prob
)

auc_score = roc_auc_score(
    y_test,
    xgb_prob
)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(f"XGBoost ROC Curve (AUC = {auc_score:.2f})")

plt.show()

print("XGBoost AUC:", auc_score)


# Feature Importance For XGBoost
print("=" * 50)
print("Feature Importance - XGBoost")
print("=" * 50)

xgb_importance = xgb_model.feature_importances_

xgb_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_importance
}).sort_values(by='Importance', ascending=False)

print("\nTOP 10 IMPORTANT FEATURES:\n")
print(xgb_importance_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=xgb_importance_df.head(10))
plt.title("Top 10 Important Features - XGBoost")
plt.tight_layout()
plt.show()



# CALIBRATION CURVE + BRIER SCORE FOR XGBOOST

xgb_prob = xgb_model.predict_proba(X_test)[:,1]

prob_true, prob_pred = calibration_curve(
    y_test,
    xgb_prob,
    n_bins=10
)

plt.figure(figsize=(6,6))

plt.plot(prob_pred, prob_true)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("Predicted Probability")

plt.ylabel("True Probability")

plt.title("XGBoost Calibration Curve")

plt.show()

brier = brier_score_loss(y_test, xgb_prob)

print("XGBoost Brier Score:", brier)


print("=" * 50)
print("LIGHTGBM CLASSIFIER")
print("=" * 50)

# LightGBM Classifier :
# -----------------------------------------------------------------------------------------

lgb_model = LGBMClassifier(
    n_estimators=100,
    random_state=42
)

lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, lgb_pred))

print("Precision:",
      precision_score(y_test, lgb_pred))

print("Recall:",
      recall_score(y_test, lgb_pred))

print("F1 Score:",
      f1_score(y_test, lgb_pred))



# Classification Report For LightGBM
print("\nClassification Report:\n")

print(classification_report(
    y_test,
    lgb_pred
))

# Confusion Matrix For LightGBM
cm = confusion_matrix(y_test, lgb_pred)

plt.figure(figsize=(5,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("LightGBM Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# (ROC + AUC) For LightGBM
lgb_prob = lgb_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    lgb_prob
)

auc_score = roc_auc_score(
    y_test,
    lgb_prob
)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(f"LightGBM ROC Curve (AUC = {auc_score:.2f})")

plt.show()

print("LightGBM AUC:", auc_score)


# Feature Importance For LightGBM
print("=" * 50)
print("Feature Importance - LightGBM")
print("=" * 50)

lgb_importance = lgb_model.feature_importances_

lgb_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_importance
}).sort_values(by='Importance', ascending=False)

print("\nTOP 10 IMPORTANT FEATURES:\n")
print(lgb_importance_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=lgb_importance_df.head(10))
plt.title("Top 10 Important Features - LightGBM")
plt.tight_layout()
plt.show()



# CALIBRATION CURVE + BRIER SCORE FOR LIGHTGBM
lgb_prob = lgb_model.predict_proba(X_test)[:,1]

prob_true, prob_pred = calibration_curve(
    y_test,
    lgb_prob,
    n_bins=10
)

plt.figure(figsize=(6,6))

plt.plot(prob_pred, prob_true)

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("Predicted Probability")

plt.ylabel("True Probability")

plt.title("LightGBM Calibration Curve")

plt.show()

brier = brier_score_loss(y_test, lgb_prob)

print("LightGBM Brier Score:", brier)


# ROC Curve + AUC :
# The ROC curve demonstrates the model’s ability to distinguish between popular and non-popular articles.
# I Did It For All Models That I Used.


# Confusion Matrix :
# The confusion matrix shows that the model correctly identified most popular and non-popular articles.
# ----------------------------------------------------------------------------------------- 


# Calibration Curve :
# Is The Model Honest , Can We Trust it? 

#-----------------------------------Final Comparsion Between Models-------------------------------#

models = {
    "Logistic Regression": lr_model,
    "KNN": knn_model,
    "SVM (Linear)": svm_linear,
    "SVM (RBF)": svm_rbf,
    "Decision Tree": tree_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

results = []

for name, model in models.items():

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC": auc
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="F1 Score", ascending=False)

print("\n===== MODEL COMPARISON =====\n")
print(results_df)


# Visual Comparison
plt.figure(figsize=(10,6))

sns.barplot(
    data=results_df,
    x="F1 Score",
    y="Model"
)

plt.title("Model Comparison (F1 Score)")
plt.show()
