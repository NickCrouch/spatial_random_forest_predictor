import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

### Generate a dataset for analysis ###

np.random.seed(42)

n = 5000

lat = np.random.uniform(30, 50, n)
lon = np.random.uniform(-120, -90, n)

elevation = np.random.normal(500, 200, n)
temperature = 25 - 0.003 * elevation + np.random.normal(0,1,n)
precipitation = np.random.normal(100, 20, n)

# spatial signal
prob = (
    0.4*(temperature > 22) +
    0.3*(precipitation > 100) +
    0.3*(elevation < 600)
)

target = np.random.binomial(1, prob)

df = pd.DataFrame({
    "lat": lat,
    "lon": lon,
    "elevation": elevation,
    "temperature": temperature,
    "precipitation": precipitation,
    "target": target
})

# df.to_csv("data/spatial_dataset.csv", index=False)

# print("Dataset saved to data/spatial_dataset.csv")

### Analyze the data ###

X = df.drop("target", axis=1)
y = df["target"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# predictions
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

# metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("ROC AUC:", roc_auc)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# feature importance
importances = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature Importance\n")
print(importances)


fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(9, 6))
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("roc_curve.png")