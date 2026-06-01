import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Read The Data :
df = pd.read_csv("data//OnlineNewsPopularity.csv")

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

#========BEFORE FEATURE ENGINEERING==========#

df_before = df.copy()

X_before = df_before.drop(columns=['shares', 'popular'])
y_before = df_before['popular']

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_before, 
    y_before, 
    test_size=0.2, 
    random_state=42
)

rf_before = RandomForestClassifier(n_estimators=100, random_state=42)

rf_before.fit(X_train_b, y_train_b)

pred_before = rf_before.predict(X_test_b)

prob_before = rf_before.predict_proba(X_test_b)[:, 1]


print("===== BEFORE FEATURE ENGINEERING =====")

print("Accuracy:", 
      accuracy_score(y_test_b, pred_before))

print("F1 Score:", 
      f1_score(y_test_b, pred_before))

print("AUC:", 
      roc_auc_score(y_test_b, prob_before))

# 1) Interaction Features
# -----------------------------------------------------------------------------------------

df['image_video_interaction'] = (
    df['num_imgs'] * df['num_videos']
)

df['keywords_shares_interaction'] = (
    df['kw_avg_avg'] * df['self_reference_avg_sharess']
)

# 2) Ratio Features
# -----------------------------------------------------------------------------------------

df['images_per_word'] = (
    df['num_imgs'] / (df['n_tokens_content'] + 1)
)

df['links_per_word'] = (
    df['num_hrefs'] / (df['n_tokens_content'] + 1)
)

# 3) Keyword Density Measures
# -----------------------------------------------------------------------------------------

df['keyword_density'] = (
    df['n_unique_tokens'] / (df['n_tokens_content'] + 1)
)

# 4) Sentiment-weighted Engagement Score
# -----------------------------------------------------------------------------------------

df['sentiment_engagement'] = (
    df['global_sentiment_polarity'] *
    df['self_reference_avg_sharess']
)

# 5) Publishing-Time Transformations
# -----------------------------------------------------------------------------------------

df['publishing_day'] = (
    df['weekday_is_monday'] * 1 +
    df['weekday_is_tuesday'] * 2 +
    df['weekday_is_wednesday'] * 3 +
    df['weekday_is_thursday'] * 4 +
    df['weekday_is_friday'] * 5 +
    df['weekday_is_saturday'] * 6 +
    df['weekday_is_sunday'] * 7
)
# It is a waek feature but I rather not to delete for some purposes.

# 6) Weekend vs Weekday Indicator
# -----------------------------------------------------------------------------------------

df['is_weekend'] = (
    df['weekday_is_saturday'] +
    df['weekday_is_sunday']
)

# 7) Article Complexity Score
# -----------------------------------------------------------------------------------------

df['article_complexity'] = (
    df['n_tokens_content'] *
    df['n_unique_tokens']
)

# 8) Log Transformations
# -----------------------------------------------------------------------------------------

df['log_num_hrefs'] = np.log1p(df['num_hrefs'])

df['log_num_imgs'] = np.log1p(df['num_imgs'])

df['log_num_videos'] = np.log1p(df['num_videos'])

# 9) Polynomial Features
# -----------------------------------------------------------------------------------------

df['tokens_squared'] = (
    df['n_tokens_content'] ** 2
)

df['shares_reference_squared'] = (
    df['self_reference_avg_sharess'] ** 2
)

#========== AFTER FEATURE ENGINEERING ============#


X_after = df.drop(columns=['shares', 'popular'])
y_after = df['popular']

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
    X_after, 
    y_after, 
    test_size=0.2, 
    random_state=42
)

rf_after = RandomForestClassifier(n_estimators=100, random_state=42)

rf_after.fit(X_train_a, y_train_a)

pred_after = rf_after.predict(X_test_a)

prob_after = rf_after.predict_proba(X_test_a)[:, 1]


print("\n===== AFTER FEATURE ENGINEERING =====")

print("Accuracy:", 
      accuracy_score(y_test_a, pred_after))

print("F1 Score:", 
      f1_score(y_test_a, pred_after))

print("AUC:", 
      roc_auc_score(y_test_a, prob_after))


#====== FINAL COMPARISON ========#

results = pd.DataFrame({
    "Metric": ["Accuracy", "F1 Score", "AUC"],
    "Before": [
        accuracy_score(y_test_b, pred_before),

        f1_score(y_test_b, pred_before),

        roc_auc_score(y_test_b, prob_before)
    ],
    "After": [
        accuracy_score(y_test_a, pred_after),

        f1_score(y_test_a, pred_after),

        roc_auc_score(y_test_a, prob_after)
    ]
})

print("\n===== FINAL COMPARISON =====")
print(results)

#========= VISUALIZATION ==========#

results.set_index("Metric").plot(kind='bar', figsize=(8,5))
plt.title("Before vs After Feature Engineering")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()

#======= FEATURE IMPORTANCE AFTER FEATURE ENGINEERING =======#
importance = rf_after.feature_importances_

feature_names = X_after.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("=" * 50)
print("Feature Importance After Feature Engineering")
print("=" * 50)

print("\nTOP 15 IMPORTANT FEATURES:\n")
print(importance_df.head(15))

#=========== PLOT TOP 15 FEATURES ============#

plt.figure(figsize=(12,7))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df.head(15)
)

plt.title("Top 15 Important Features After Feature Engineering")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()

plt.show()

# Note #1:
# In some cases, engineered features may not appear in the Top 15 ,
# but it can show up if we do Top 30 , suggesting that their impact
# is present and our Feature Engeineering is working.

# Note #2:
# We Choose "Random Forest Model" because he can deal with large data , he less affected with
# noise data and he is strong with Feature Engeineering .