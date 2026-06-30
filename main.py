import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
data = load_breast_cancer()

# X = features (input), y = labels (what we want to predict)
X = data.data
y = data.target

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dataset loaded successfully!")
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Train k-NN model
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

# Print accuracy scores
print("\n--- Model Results ---")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_predictions):.4f}")
print(f"k-NN Accuracy:          {accuracy_score(y_test, knn_predictions):.4f}")

# Print confusion matrices
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nk-NN Confusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))

# CONCLUSION
# =============================================================
# Both Random Forest and k-NN achieved around 96% accuracy on
# the Breast Cancer dataset. However, Random Forest performed
# slightly better (96.49% vs 95.61%).
#
# Looking at the confusion matrices, Random Forest missed only
# 3 malignant tumors (False Negatives), while k-NN missed 5.
# In medical diagnosis, missing a malignant tumor is far more
# dangerous than a false alarm, so Random Forest is the safer
# and better model for this specific dataset.
#
#The reason Random Forest performed well could be that it uses
#hundreds of decision trees and takes a vote for the output,
#thereby being less affected by noise in the data set.
#k-NN is affected by the units of measurement used for the
#different attributes and without applying feature scaling,
#it performed slightly lower.