import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data with validation
print("Loading Fashion-MNIST dataset...")
try:
    train_df = pd.read_csv("fashion-mnist_train.csv")
    test_df = pd.read_csv("fashion-mnist_test.csv")
    print("✓ Data loaded successfully")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure fashion-mnist_train.csv and fashion-mnist_test.csv are in the current directory")
    exit(1)

# Define class names for better interpretability
class_names = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

# 2. Preprocess Data with normalization
X_train = train_df.drop(columns=["label"]).values
y_train = train_df["label"].values
X_test = test_df.drop(columns=["label"]).values
y_test = test_df["label"].values

# Normalize pixel values to [0,1] range for better tree performance
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"Shape of training data: {X_train.shape}")
print(f"Shape of test data: {X_test.shape}")
print(f"Unique classes: {sorted(np.unique(y_train))}")
print(f"Class distribution in training set:")
for i, count in enumerate(np.bincount(y_train)):
    print(f"  {i} ({class_names[i]}): {count:,} samples ({count/len(y_train)*100:.1f}%)")

# 3. Train Baseline Model with Cross-Validation
print("\n" + "="*50)
print("BASELINE MODEL EVALUATION")
print("="*50)

start_time = time.time()
clf = DecisionTreeClassifier(random_state=42)

# Cross-validation for more reliable performance estimation
cv_scores = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1)
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

# Fit and evaluate on test set
clf.fit(X_train, y_train)
y_pred_base = clf.predict(X_test)
base_accuracy = accuracy_score(y_test, y_pred_base)
base_f1 = f1_score(y_test, y_pred_base, average='macro')

print(f"Test Accuracy: {base_accuracy:.4f}")
print(f"Test Macro F1: {base_f1:.4f}")
print(f"Training time: {time.time() - start_time:.2f} seconds")

# 4. Feature Selection (Optional - reduces overfitting)
print("\n" + "="*50)
print("FEATURE SELECTION")
print("="*50)

# Select top k features using chi-squared test
k_features = 400  # Reduce from 784 to 400 most informative pixels
selector = SelectKBest(score_func=chi2, k=k_features)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
print(f"Selected {k_features} most informative features out of {X_train.shape[1]}")

# 5. Hyperparameter Tuning with Enhanced Grid
print("\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)

param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [8, 12, 16, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
    "class_weight": [None, "balanced"]
}

print("Starting grid search (this may take a while)...")
start_time = time.time()

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1_macro",  # Use F1 for better handling of class imbalance
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

# Use feature-selected data for tuning
grid.fit(X_train_selected, y_train)
best = grid.best_estimator_

print(f"Grid search completed in {time.time() - start_time:.2f} seconds")
print("Best Parameters:", grid.best_params_)
print(f"Best CV F1-Macro Score: {grid.best_score_:.4f}")

# 6. Evaluate Tuned Model
print("\n" + "="*50)
print("TUNED MODEL EVALUATION")
print("="*50)

y_pred = best.predict(X_test_selected)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"Tuned Accuracy: {acc:.4f}")
print(f"Tuned Macro F1: {f1:.4f}")
print(f"Tuned Weighted F1: {f1_weighted:.4f}")
print(f"Improvement over baseline: {acc - base_accuracy:.4f} ({((acc/base_accuracy - 1)*100):+.1f}%)")

print("\nDetailed Classification Report:")
target_names = [f"{i}: {class_names[i]}" for i in range(10)]
print(classification_report(y_test, y_pred, target_names=target_names))

# Per-class accuracy
print("\nPer-Class Accuracy:")
cm = confusion_matrix(y_test, y_pred)
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, acc_class in enumerate(class_accuracy):
    print(f"  {class_names[i]}: {acc_class:.3f}")

# 7. Enhanced Confusion Matrix Visualization
print("\n" + "="*50)
print("VISUALIZATION")
print("="*50)

# Confusion Matrix with class names
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))

# Normalize confusion matrix for better visualization
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create annotation for both count and percentage
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_norm[i, j]
        if c == 0:
            annot[i, j] = ''
        else:
            annot[i, j] = f'{c}\n({p:.2%})'

# Create the heatmap
sns.heatmap(cm_norm, annot=annot, fmt='', cmap="Blues",
            xticklabels=[class_names[i] for i in range(10)],
            yticklabels=[class_names[i] for i in range(10)],
            cbar_kws={'label': 'Normalized Frequency'})

plt.title("Confusion Matrix - Fashion-MNIST Decision Tree\n(Normalized with counts and percentages)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 8. Enhanced Feature Importance Analysis
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Get feature importance (mapped back to original features)
selected_features = selector.get_support(indices=True)
importance_selected = best.feature_importances_

# Create full importance array
importance_full = np.zeros(X_train.shape[1])
importance_full[selected_features] = importance_selected

# Top important pixels
top_indices = np.argsort(importance_full)[::-1][:30]
print("Top 30 important pixel positions:")
for i, idx in enumerate(top_indices[:10]):
    row, col = divmod(idx, 28)
    print(f"  {i+1:2d}. Pixel {idx:3d} (row {row:2d}, col {col:2d}): {importance_full[idx]:.6f}")

# Visualize importance as heatmap
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original importance heatmap
importance_img = importance_full.reshape(28, 28)
im1 = axes[0].imshow(importance_img, cmap="hot", interpolation='nearest')
axes[0].set_title("Feature Importance Heatmap\n(All Pixels)")
axes[0].axis('off')
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Selected features heatmap
selected_img = np.zeros(784)
selected_img[selected_features] = 1
selected_img = selected_img.reshape(28, 28)
im2 = axes[1].imshow(selected_img, cmap="Greens", interpolation='nearest')
axes[1].set_title(f"Selected Features\n({k_features} pixels)")
axes[1].axis('off')
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# Combined view
combined_img = importance_img * selected_img
im3 = axes[2].imshow(combined_img, cmap="hot", interpolation='nearest')
axes[2].set_title("Importance × Selection\n(Final Features)")
axes[2].axis('off')
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# 9. Visualize a small tree for interpretation
print("\nCreating interpretable tree visualization...")
viz_clf = DecisionTreeClassifier(
    max_depth=3, min_samples_split=20, min_samples_leaf=10,
    random_state=42, class_weight='balanced'
)
viz_clf.fit(X_train_selected, y_train)

plt.figure(figsize=(25, 15))
plot_tree(
    viz_clf,
    filled=True,
    feature_names=[f"px{selected_features[i]}" for i in range(len(selected_features))],
    class_names=[class_names[i] for i in range(10)],
    rounded=True,
    fontsize=10,
    proportion=True,
    impurity=True
)
plt.title("Shallow Decision Tree for Interpretation\n(Max Depth=3, Selected Features Only)")
plt.tight_layout()
plt.show()

# 10. Model Summary
print("\n" + "="*50)
print("FINAL MODEL SUMMARY")
print("="*50)
print(f"Dataset: Fashion-MNIST ({X_train.shape[0]:,} train, {X_test.shape[0]:,} test)")
print(f"Original features: {X_train.shape[1]}")
print(f"Selected features: {k_features}")
print(f"Final test accuracy: {acc:.4f}")
print(f"Final test F1-macro: {f1:.4f}")
print(f"Best hyperparameters: {grid.best_params_}")
print(f"Tree depth: {best.tree_.max_depth}")
print(f"Number of leaves: {best.tree_.n_leaves}")
print(f"Total nodes: {best.tree_.node_count}")