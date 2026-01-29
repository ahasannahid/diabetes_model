# Common imports for Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# Plot settings
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['axes.grid'] = True





df = pd.read_csv("diabetes.csv")

# print("Dataset Shape (rows, columns):", df.shape)

# Display DF
# print(df)

features = df.drop("Outcome", axis=1)
Q1 = features.quantile(0.25)
Q3 = features.quantile(0.75)
IQR = Q3 - Q1

outlier_mask = (features < (Q1 - 1.5 * IQR)) | (features > (Q3 + 1.5 * IQR))

outlier_count = outlier_mask.sum().sort_values(ascending=False)


zero_count = (df == 0).sum()

zero_count.sort_values(ascending=False)

# which column have 0
zero_as_missing = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

# 0 replace with NaN
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

# nan replace with median
for col in zero_as_missing:
    df[col] = df[col].fillna(df[col].median())

# print(df[zero_as_missing].isna().sum())

# features & target
X = df.drop("Outcome", axis=1)

# scaler
scaler = StandardScaler()


X_scaled = scaler.fit_transform(X)


X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


df_scaled = X_scaled.copy()
df_scaled["Outcome"] = df["Outcome"]


df["BMI_category"] = pd.cut(
    df_scaled["BMI"],
    bins=[0, 18.5, 25, 30, 100],
    labels=[0, 1, 2, 3]
)
df["Age_group"] = pd.cut(
    df_scaled["Age"],
    bins=[20, 30, 45, 60, 100],
    labels=[0, 1, 2, 3]
)
df_scaled["Glucose_BMI"] = df["Glucose"] * df["BMI"]
df_scaled["High_Insulin"] = (df["Insulin"] > df["Insulin"].median()).astype(int)


df_scaled["Outcome"].value_counts()

df_scaled["Outcome"].value_counts(normalize=True) * 100

class_summary = df_scaled["Outcome"].value_counts().to_frame(name="count")
class_summary["percent"] = (class_summary["count"] / len(df_scaled) * 100).round(2)

df_scaled["Outcome"].value_counts().plot(kind="bar")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.show()

pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

X = df_scaled.drop("Outcome", axis=1)
y = df_scaled["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)



X = df_scaled.drop("Outcome", axis=1)
y = df_scaled["Outcome"]


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

# print("Cross-validation accuracies:", cv_scores)
# print("Mean accuracy:", cv_scores.mean())
# print("Standard deviation:", cv_scores.std())


X = df_scaled.drop("Outcome", axis=1)
y = df_scaled["Outcome"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs", "liblinear"],
    "model__class_weight": [None, "balanced"]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

grid.fit(X, y)

# print("Best Parameters:", grid.best_params_)
# print("Best CV Accuracy:", grid.best_score_)

results = pd.DataFrame(grid.cv_results_)


show_cols = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
results[show_cols].sort_values("rank_test_score").head(15)

best_model = grid.best_estimator_
# print("Best Hyperparameters:")
# print(grid.best_params_)

# print("\nBest Cross-Validated Accuracy:")
# print(grid.best_score_)


y_pred = best_model.predict(X_test)


# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))


y_prob = best_model.predict_proba(X_test)[:, 1]
# print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# save the model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved successfully!")