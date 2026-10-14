# ============================================================
# Tutorial Quality Scorer — WnCC Convenor Assignment 2026-27
# ============================================================
# This script:
#   1. Loads a synthetic YouTube tutorial dataset
#   2. Engineers features that signal tutorial quality
#   3. Trains a Random Forest model to predict quality scores
#   4. Outputs feature importances and top 10 best tutorials
#   5. Visualises results and explains individual video scores
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: Load the dataset
# ─────────────────────────────────────────────

df = pd.read_csv('tutorials.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total tutorials loaded : {len(df)}")
print(f"Quality Score range    : {df['Actual Quality Score'].min()} - {df['Actual Quality Score'].max()}")
print(f"Avg duration (minutes) : {df['Duration Minutes'].mean():.1f}")
print(f"Avg view count         : {df['View Count'].mean():,.0f}")
print()

# ─────────────────────────────────────────────
# STEP 2: Feature Engineering
# ─────────────────────────────────────────────

df['like_to_view_ratio'] = df['Like Count'] / df['View Count']
df['engagement_rate'] = (df['Like Count'] + df['Comment Count']) / df['View Count']
df['comment_to_view_ratio'] = df['Comment Count'] / df['View Count']

CLICKBAIT_PATTERNS = [
    'in 5 mins', 'in 3 minutes', 'in 10 minutes', 'overnight',
    'secrets', 'you need', 'stop using', 'is dead', '$100k',
    '10x', 'make money', 'quit', 'only guide', 'fastest',
    'in 7 minutes', 'in 8 minutes', 'in 12 minutes', 'in 6 minutes',
    'chatgpt writes', 'no bs', 'impossible', 'super easy',
    'crash guide', 'fast guide', "they don't tell you",
    '\U0001f92f', '\U0001f525', '\U0001f480', '\U0001f911',
    '\U0001f631', '\U0001f92b', '\U0001f6d1', '\U0001f916',
    '\U0001f60e', '\U0001f632', '\U0001f680'
]

def compute_clickbait_score(title):
    title_lower = title.lower()
    return sum(1 for pattern in CLICKBAIT_PATTERNS if pattern in title_lower)

df['clickbait_score'] = df['Video Title'].apply(compute_clickbait_score)
df['desc_length'] = df['Video Description'].apply(len)

min_dur = df['Duration Minutes'].min()
max_dur = df['Duration Minutes'].max()
df['duration_normalized'] = (df['Duration Minutes'] - min_dur) / (max_dur - min_dur)
df['view_to_subscriber_ratio'] = df['View Count'] / df['Channel Subscriber Count']

print("=" * 60)
print("FEATURE ENGINEERING - Sample (first 5 rows)")
print("=" * 60)
print(df[['Video Title', 'like_to_view_ratio', 'clickbait_score',
          'engagement_rate', 'duration_normalized']].head().to_string())
print()

# ─────────────────────────────────────────────
# STEP 3: Prepare data
# ─────────────────────────────────────────────

FEATURES = [
    'like_to_view_ratio',
    'engagement_rate',
    'comment_to_view_ratio',
    'clickbait_score',
    'desc_length',
    'duration_normalized',
    'view_to_subscriber_ratio'
]

X = df[FEATURES]
y = df['Actual Quality Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# STEP 4: Train Models
# ─────────────────────────────────────────────

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_r2   = r2_score(y_test, rf_pred)

rf_cv_scores = cross_val_score(rf_model, X, y, cv=5,
                                scoring='neg_mean_absolute_error')
rf_cv_mae = -rf_cv_scores.mean()

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_r2   = r2_score(y_test, lr_pred)

print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"{'Model':<25} {'MAE':>6} {'R2':>8} {'CV MAE (5-fold)':>16}")
print("-" * 60)
print(f"{'Random Forest':<25} {rf_mae:>6.2f} {rf_r2:>8.4f} {rf_cv_mae:>16.2f}")
print(f"{'Linear Regression':<25} {lr_mae:>6.2f} {lr_r2:>8.4f} {'N/A':>16}")
print()
print("MAE    = Mean Absolute Error (lower is better)")
print("R2     = Variance explained (closer to 1.0 is better)")
print("CV MAE = Cross-validated MAE (more reliable than single split)")
print()

# ─────────────────────────────────────────────
# STEP 5: Feature Importances
# ─────────────────────────────────────────────

importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

print("=" * 60)
print("FEATURE IMPORTANCES")
print("=" * 60)
for feat, score in importances.items():
    bar = 'X' * int(score * 50)
    print(f"{feat:<30} {score:.4f}  {bar}")

top_feature = importances.idxmax()
corr = df[top_feature].corr(df['Actual Quality Score'])
direction = "higher = better quality" if corr > 0 else "higher = worse quality"
print(f"\n>>> Top feature: '{top_feature}'")
print(f"    Pearson r = {corr:.4f} ({direction})")
print()

# ─────────────────────────────────────────────
# STEP 6: Visualisation
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

importances.sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Feature Importances\n(What Predicts Tutorial Quality?)',
                   fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance Score')

all_pred = rf_model.predict(X)
axes[1].scatter(y, all_pred, alpha=0.5, color='steelblue', s=30)
axes[1].plot([y.min(), y.max()], [y.min(), y.max()],
             'r--', linewidth=1.5, label='Perfect prediction')
axes[1].set_xlabel('Actual Quality Score')
axes[1].set_ylabel('Predicted Quality Score')
axes[1].set_title('Actual vs Predicted Quality Score\n(Random Forest)',
                   fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as: model_analysis.png\n")

# ─────────────────────────────────────────────
# STEP 7: Predict all tutorials
# ─────────────────────────────────────────────

df['Predicted_Score'] = rf_model.predict(X).round(2)
df['Predicted_Score'] = df['Predicted_Score'].clip(0, 100)

# ─────────────────────────────────────────────
# STEP 8: Top 10 and Bottom 5
# ─────────────────────────────────────────────

top10 = df.nlargest(10, 'Predicted_Score')[
    ['Video Title', 'Duration Minutes', 'like_to_view_ratio',
     'clickbait_score', 'Actual Quality Score', 'Predicted_Score']
].reset_index(drop=True)
top10.index += 1

print("=" * 60)
print("TOP 10 BEST TUTORIALS (by Predicted Score)")
print("=" * 60)
print(top10.to_string())
print()

bottom5 = df.nsmallest(5, 'Predicted_Score')[
    ['Video Title', 'clickbait_score', 'like_to_view_ratio',
     'Actual Quality Score', 'Predicted_Score']
].reset_index(drop=True)
bottom5.index += 1

print("=" * 60)
print("BOTTOM 5 (Worst / Most Clickbait)")
print("=" * 60)
print(bottom5.to_string())
print()

# ─────────────────────────────────────────────
# STEP 9: Clickbait Explainer
# ─────────────────────────────────────────────

def explain_video(title, dataframe):
    """
    Prints a human-readable breakdown of why a video scored the way it did.
    Makes the model interpretable beyond just a number.
    """
    row = dataframe[dataframe['Video Title'] == title]
    if row.empty:
        print(f"  Video not found: '{title}'\n")
        return

    row = row.iloc[0]
    score = row['Predicted_Score']
    reasons = []

    if row['clickbait_score'] >= 2:
        reasons.append(f"  [!] Title has {int(row['clickbait_score'])} clickbait signals "
                       f"(emojis, 'in X minutes', sensational phrasing)")
    elif row['clickbait_score'] == 1:
        reasons.append(f"  [!] Title has 1 mild clickbait signal")

    if row['like_to_view_ratio'] < 0.015:
        reasons.append(f"  [!] Like ratio = {row['like_to_view_ratio']:.4f} "
                       f"(viewers felt misled - low satisfaction)")
    elif row['like_to_view_ratio'] > 0.04:
        reasons.append(f"  [+] Like ratio = {row['like_to_view_ratio']:.4f} "
                       f"(viewers found it genuinely useful)")

    if row['duration_normalized'] < 0.08:
        reasons.append(f"  [!] Duration = {int(row['Duration Minutes'])} min "
                       f"(too short for meaningful depth)")
    elif row['duration_normalized'] > 0.5:
        reasons.append(f"  [+] Duration = {int(row['Duration Minutes'])} min "
                       f"(sufficient coverage and depth)")

    if row['engagement_rate'] < 0.012:
        reasons.append(f"  [!] Engagement = {row['engagement_rate']:.4f} "
                       f"(passive viewers - little interaction)")
    elif row['engagement_rate'] > 0.04:
        reasons.append(f"  [+] Engagement = {row['engagement_rate']:.4f} "
                       f"(active audience - liked and commented)")

    if not reasons:
        reasons.append("  [+] No major quality issues detected")

    verdict = ("POOR" if score < 40 else "AVERAGE" if score < 70 else "GOOD")

    print(f"  Title   : {title}")
    print(f"  Score   : {score:.1f}/100  [{verdict}]")
    print(f"  Signals :")
    for r in reasons:
        print(r)
    print()


print("=" * 60)
print("CLICKBAIT EXPLAINER - Individual Video Analysis")
print("=" * 60)

worst = df.nsmallest(1, 'Predicted_Score')['Video Title'].values[0]
best  = df.nlargest(1,  'Predicted_Score')['Video Title'].values[0]
mid   = df.iloc[(df['Predicted_Score'] - 55).abs().argsort()[:1]]['Video Title'].values[0]

print("\n[WORST VIDEO]")
explain_video(worst, df)
print("[MID-TIER VIDEO]")
explain_video(mid, df)
print("[BEST VIDEO]")
explain_video(best, df)

# ─────────────────────────────────────────────
# STEP 10: Save output CSV
# ─────────────────────────────────────────────

output = df[['Video Title', 'Duration Minutes', 'Actual Quality Score',
             'Predicted_Score', 'clickbait_score',
             'like_to_view_ratio', 'engagement_rate']].copy()
output = output.sort_values('Predicted_Score', ascending=False).reset_index(drop=True)
output.index += 1
output.index.name = 'Rank'
output.to_csv('scored_tutorials.csv')

print("=" * 60)
print("Output saved : scored_tutorials.csv")
print("Plot saved   : model_analysis.png")
print("=" * 60)

print(f"""
HOW THE MODEL PENALISES CLICKBAIT
===================================
Clickbait tutorials share a consistent data pattern:
  - High View Count     -> flashy thumbnail drives clicks
  - Low Like Ratio      -> viewers feel misled, don't engage
  - Short Duration      -> not enough time to actually teach
  - High Clickbait Score -> sensational title with emojis/hype

Random Forest learns these correlations across 100 decision trees.
Each tree splits on a feature threshold (e.g. duration < 15 min?)
and votes on a quality score. Final = average of all 100 votes.

Top feature     : '{top_feature}' (importance = {importances[top_feature]:.4f})
CV MAE (5-fold) : {rf_cv_mae:.2f} points (consistent across different splits)
""")
