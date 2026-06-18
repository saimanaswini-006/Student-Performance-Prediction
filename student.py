import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ── GENERATE SAMPLE DATA (replace with your real dataset) ────────────────
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'study_hours_per_day': np.random.uniform(1, 10, n),
    'attendance_percent': np.random.uniform(50, 100, n),
    'sleep_hours': np.random.uniform(4, 9, n),
    'previous_score': np.random.uniform(40, 100, n),
    'extracurricular': np.random.choice([0, 1], n),
    'parental_education': np.random.choice(['high_school', 'bachelor', 'master'], n),
    'internet_access': np.random.choice([0, 1], n),
    'gender': np.random.choice(['male', 'female'], n),
})

# Target: exam score
data['exam_score'] = (
    data['study_hours_per_day'] * 4.5 +
    data['attendance_percent'] * 0.3 +
    data['previous_score'] * 0.4 +
    data['sleep_hours'] * 1.2 +
    np.random.normal(0, 5, n)
).clip(0, 100)

print("Dataset Shape:", data.shape)
print(data.head())

# ── EDA ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(data['exam_score'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of Exam Scores')
axes[0].set_xlabel('Exam Score')

axes[1].scatter(data['study_hours_per_day'], data['exam_score'], alpha=0.4, color='coral')
axes[1].set_title('Study Hours vs Exam Score')
axes[1].set_xlabel('Study Hours/Day')
axes[1].set_ylabel('Exam Score')

axes[2].scatter(data['attendance_percent'], data['exam_score'], alpha=0.4, color='green')
axes[2].set_title('Attendance vs Exam Score')
axes[2].set_xlabel('Attendance %')

plt.tight_layout()
plt.savefig('eda_plots.png')
plt.show()

# ── PREPROCESSING ─────────────────────────────────────────────────────────
le = LabelEncoder()
data['parental_education'] = le.fit_transform(data['parental_education'])
data['gender'] = le.fit_transform(data['gender'])

X = data.drop('exam_score', axis=1)
y = data['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── TRAIN MULTIPLE MODELS ─────────────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05,
                            max_depth=6, random_state=42, verbosity=0)
}

results = {}
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
    print(f"{name:20s} → R²: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

# ── FEATURE IMPORTANCE (XGBoost) ──────────────────────────────────────────
xgb_model = models['XGBoost']
importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importance.plot(kind='barh', color='steelblue')
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# ── CORRELATION HEATMAP ───────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# ── PREDICT FOR NEW STUDENT ───────────────────────────────────────────────
def predict_score(student_data: dict, model, feature_cols):
    df = pd.DataFrame([student_data])[feature_cols]
    score = model.predict(df)[0]
    print(f"\nPredicted Exam Score: {score:.1f} / 100")
    return score

sample_student = {
    'study_hours_per_day': 6,
    'attendance_percent': 85,
    'sleep_hours': 7,
    'previous_score': 75,
    'extracurricular': 1,
    'parental_education': 1,
    'internet_access': 1,
    'gender': 0
}

predict_score(sample_student, xgb_model, X.columns)