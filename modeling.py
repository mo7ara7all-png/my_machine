import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

df = pd.read_csv("Data//OnlineNewsPopularity.csv")
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
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =====================================================
# DEFINE CLASSIFICATION MODELS
# =====================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    # "SVM Linear": SVC(
    #     kernel='linear',
    #     probability=True,
    #     random_state=42
    # ),

    # "SVM RBF": SVC(
    #     kernel='rbf',
    #     probability=True,
    #     random_state=42
    # ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='logloss'
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=100,
        random_state=42
    )
}

# =====================================================
# 1. VALIDATION VS TRAINING PERFORMANCE
# =====================================================

print("\n================ VALIDATION VS TRAINING PERFORMANCE ================\n")

comparison_results = []

for name, model in models.items():

    print("=" * 50)
    print(name)
    print("=" * 50)

    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)

    test_pred = model.predict(X_test_scaled)

    train_f1 = f1_score(y_train, train_pred)

    test_f1 = f1_score(y_test, test_pred)

    train_accuracy = accuracy_score(y_train, train_pred)

    test_accuracy = accuracy_score(y_test, test_pred)

    if hasattr(model, "predict_proba"):
        test_prob = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_prob)
    else:
        test_auc = np.nan

    gap = train_f1 - test_f1

    comparison_results.append({
        "Model": name,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Train F1": train_f1,
        "Test F1": test_f1,
        "AUC": test_auc,
        "Train-Test Gap": gap
    })

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("Train F1:", train_f1)
    print("Test F1:", test_f1)
    print("AUC:", test_auc)
    print("Train-Test Gap:", gap)

comparison_df = pd.DataFrame(comparison_results)

comparison_df = comparison_df.sort_values(by="Test F1", ascending=False)

print("\n================ FINAL MODEL COMPARISON TABLE ================\n")
print(comparison_df)

# =====================================================
# VISUAL COMPARISON - TEST F1 SCORE
# =====================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=comparison_df,
    x="Test F1",
    y="Model"
)

plt.title("Model Comparison Based on Test F1 Score")
plt.xlabel("Test F1 Score")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# =====================================================
# VISUAL COMPARISON - AUC
# =====================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=comparison_df,
    x="AUC",
    y="Model"
)

plt.title("Model Comparison Based on AUC")
plt.xlabel("AUC Score")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# =====================================================
# TRAIN VS TEST F1 COMPARISON
# =====================================================

train_test_df = comparison_df[["Model", "Train F1", "Test F1"]]

train_test_df = train_test_df.melt(
    id_vars="Model",
    value_vars=["Train F1", "Test F1"],
    var_name="Dataset",
    value_name="F1 Score"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=train_test_df,
    x="Model",
    y="F1 Score",
    hue="Dataset"
)

plt.title("Training vs Testing F1 Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =====================================================
# 2. BIAS-VARIANCE DISCUSSION TABLE
# =====================================================

print("\n================ BIAS-VARIANCE ANALYSIS ================\n")

bias_variance_results = []

for index, row in comparison_df.iterrows():

    model_name = row["Model"]

    train_f1 = row["Train F1"]

    test_f1 = row["Test F1"]

    gap = row["Train-Test Gap"]

    if train_f1 < 0.65 and test_f1 < 0.65:
        diagnosis = "High Bias / Underfitting"

    elif gap > 0.10:
        diagnosis = "High Variance / Overfitting"

    else:
        diagnosis = "Good Generalization"

    bias_variance_results.append({
        "Model": model_name,
        "Train F1": train_f1,
        "Test F1": test_f1,
        "Gap": gap,
        "Diagnosis": diagnosis
    })

bias_variance_df = pd.DataFrame(bias_variance_results)

print(bias_variance_df)

# =====================================================
# 3. LEARNING CURVES
# =====================================================

print("\n================ LEARNING CURVES ================\n")

selected_models_for_learning_curve = {
    "Logistic Regression": models["Logistic Regression"],
    "Random Forest": models["Random Forest"],
    "XGBoost": models["XGBoost"]
}

for name, model in selected_models_for_learning_curve.items():

    print("Creating Learning Curve for:", name)

    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X_train_scaled,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)

    validation_mean = np.mean(validation_scores, axis=1)

    plt.figure(figsize=(8, 6))

    plt.plot(
        train_sizes,
        train_mean,
        marker='o',
        label="Training F1"
    )

    plt.plot(
        train_sizes,
        validation_mean,
        marker='o',
        label="Validation F1"
    )

    plt.title("Learning Curve - " + name)

    plt.xlabel("Training Set Size")

    plt.ylabel("F1 Score")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()

# =====================================================
# 4. STATISTICAL COMPARISON BETWEEN MODELS
# =====================================================

print("\n================ STATISTICAL COMPARISON BETWEEN MODELS ================\n")

cv_results = []

model_cv_scores = {}

for name, model in models.items():

    print("Cross-validating:", name)

    scores = cross_validate(
        estimator=model,
        X=X_train_scaled,
        y=y_train,
        cv=5,
        scoring={
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'auc': 'roc_auc'
        },
        n_jobs=-1
    )
    model_cv_scores[name] = scores['test_f1']

    cv_results.append({
        "Model": name,
        "Mean Accuracy": np.mean(scores['test_accuracy']),
        "Mean Precision": np.mean(scores['test_precision']),
        "Mean Recall": np.mean(scores['test_recall']),
        "Mean F1": np.mean(scores['test_f1']),
        "Std F1": np.std(scores['test_f1']),
        "Mean AUC": np.mean(scores['test_auc'])
    })

cv_results_df = pd.DataFrame(cv_results)

cv_results_df = cv_results_df.sort_values(by="Mean F1", ascending=False)

print("\n================ CROSS-VALIDATION RESULTS ================\n")
print(cv_results_df)

# =====================================================
# VISUALIZE CROSS-VALIDATION F1
# =====================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=cv_results_df,
    x="Mean F1",
    y="Model"
)

plt.title("Cross-Validation Model Comparison - Mean F1")
plt.xlabel("Mean F1 Score")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# =====================================================
# 5. STATISTICAL SIGNIFICANCE EXPERIMENT
# =====================================================

print("\n================ STATISTICAL SIGNIFICANCE TEST ================\n")

best_model_name = cv_results_df.iloc[0]["Model"]

second_model_name = cv_results_df.iloc[1]["Model"]

best_scores = model_cv_scores[best_model_name]

second_scores = model_cv_scores[second_model_name]

print("Best Model:", best_model_name)
print("Second Best Model:", second_model_name)

print("\nBest Model CV F1 Scores:")
print(best_scores)

print("\nSecond Best Model CV F1 Scores:")
print(second_scores)


# Paired t-test
t_stat, p_value = ttest_rel(best_scores, second_scores)

print("\nPaired t-test result:")
print("t-statistic:", t_stat)
print("p-value:", p_value)


# Wilcoxon signed-rank test
wilcoxon_stat, wilcoxon_p_value = wilcoxon(best_scores, second_scores)

print("\nWilcoxon signed-rank test result:")
print("statistic:", wilcoxon_stat)
print("p-value:", wilcoxon_p_value)

# =====================================================
# INTERPRET STATISTICAL TEST
# =====================================================

alpha = 0.05

print("\n================ STATISTICAL TEST INTERPRETATION ================\n")

if p_value < alpha:
    print("Paired t-test: The difference between the two models is statistically significant.")
else:
    print("Paired t-test: The difference between the two models is NOT statistically significant.")

if wilcoxon_p_value < alpha:
    print("Wilcoxon test: The difference between the two models is statistically significant.")
else:
    print("Wilcoxon test: The difference between the two models is NOT statistically significant.")

# =====================================================
# FINAL MODEL SELECTION
# =====================================================

print("\n================ FINAL MODEL SELECTION ================\n")

print("Recommended Model Based on Cross-Validation F1:", best_model_name)

print("\nReason:")
print("- It achieved the highest mean F1 score during cross-validation.")
print("- The train-test gap was checked to detect overfitting.")
print("- Learning curves were used to study whether the model improves with more data.")
print("- Statistical tests were used to check whether the difference from the second-best model is meaningful.")