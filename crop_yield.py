import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ─── 1. LOAD DATA ────────────────────────────────────────────
df = pd.read_csv("crop_production.csv")
print("Shape:", df.shape)
print(df.head())

# ─── 2. CLEAN DATA ───────────────────────────────────────────
df.dropna(inplace=True)
df = df[df['Production'] > 0]
df = df[df['Area'] > 0]

# ─── 3. CREATE YIELD PER HECTARE ─────────────────────────────
df['Yield_per_hectare'] = (df['Production'] / df['Area']) * 1000

# Remove extreme outliers
df = df[df['Yield_per_hectare'] < df['Yield_per_hectare'].quantile(0.99)]
df = df[df['Yield_per_hectare'] > df['Yield_per_hectare'].quantile(0.01)]

print("\nYield per hectare stats:")
print(df['Yield_per_hectare'].describe())

# ─── 4. ENCODE CATEGORICAL COLUMNS ──────────────────────────
le = LabelEncoder()
df['State_Name']    = le.fit_transform(df['State_Name'])
df['District_Name'] = le.fit_transform(df['District_Name'])
df['Crop']          = le.fit_transform(df['Crop'])
df['Season']        = le.fit_transform(df['Season'])

# ─── 5. EDA PLOTS ────────────────────────────────────────────
plt.figure(figsize=(10, 4))
sns.histplot(df['Yield_per_hectare'], bins=50, kde=True, color='steelblue')
plt.title("Yield per Hectare Distribution")
plt.xlabel("kg / hectare")
plt.savefig("production_dist.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation.png")
plt.show()

# ─── 6. PREPARE FEATURES ─────────────────────────────────────
X = df.drop(['Production', 'Yield_per_hectare'], axis=1)
y = df['Yield_per_hectare']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── 7. TRAIN & COMPARE MODELS ───────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost":           XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "R2":   round(r2_score(y_test, preds), 3),
        "RMSE": round(mean_squared_error(y_test, preds)**0.5, 2),
        "MAE":  round(mean_absolute_error(y_test, preds), 2)
    }
    print(f"\n{name}")
    print(f"  R²   : {results[name]['R2']}")
    print(f"  RMSE : {results[name]['RMSE']}")
    print(f"  MAE  : {results[name]['MAE']}")

# ─── 8. FEATURE IMPORTANCE ───────────────────────────────────
best_model = models["Random Forest"]
importances = pd.Series(best_model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8, 5), color='steelblue')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ─── 9. SAVE BEST MODEL ──────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("\nModel saved as model.pkl")