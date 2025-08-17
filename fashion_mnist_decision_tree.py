import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

# 2. Preprocess Data
X_train = train_df.drop(columns=["label"]).values
y_train = train_df["label"].values
X_test = test_df.drop(columns=["label"]).values
y_test = test_df["label"].values

print(f"Shape of training data: {X_train.shape}")
print(f"Shape of test data: {X_test.shape}")

# 3. Train Model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred_base = clf.predict(X_test)
print(f"Accuracy of the Decision Tree classifier: {accuracy_score(y_test, y_pred_base):.2f}")
print(f"F1 score of the Decision Tree classifier: {f1_score(y_test, y_pred_base, average='macro'):.2f}")

# 3. Hyperparameter Tuning
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 2, 5]
}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best Params:", grid.best_params_)

# 4. Evaluate Tuned Model
y_pred = best.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
print(f"Tuned Accuracy: {acc:.4f}")
print(f"Tuned Macro F1: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#  5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Fashion-MNIST Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# 6. Visualize a small/pruned tree (optional)
# Full tree with 784 features is huge; fit a small tree just for plotting.
viz_clf = DecisionTreeClassifier(
    max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42
)
viz_clf.fit(X_train, y_train)
plt.figure(figsize=(20, 10))
plot_tree(
    viz_clf,
    filled=True,
    feature_names=[f"px{i}" for i in range(X_train.shape[1])],
    class_names=[str(i) for i in range(10)],
    rounded=True,
    fontsize=8
)
plt.title("Shallow Decision Tree (Visualization Only)")
plt.tight_layout()
plt.show()

# 7) Top pixel importance
importance = best.feature_importances_
idx = np.argsort(importance)[::-1][:20]
print("Top 20 important pixel indices:", idx.tolist())
print("Top 20 importance:", importance[idx].round(6).tolist())

# Optional: visualize importance as a 28x28 heatmap
try:
    importance_img = importance.reshape(28, 28)
    plt.figure(figsize=(4,4))
    plt.imshow(importance_img, cmap="hot")
    plt.title("Pixel Importance (Tuned Tree)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Skipping importance heatmap:", e)