import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Load the data
diabetes_data = pd.read_csv('./diabetes.csv')

# Dataset Details
dataset_info = {
    "shape": diabetes_data.shape,
    "columns": diabetes_data.columns.tolist(),
    "missing_values": diabetes_data.isnull().sum().to_dict(),
    "outcome_counts": diabetes_data['Outcome'].value_counts().to_dict()
}

# Separate features and target
features = diabetes_data.drop(columns='Outcome', axis=1)
target = diabetes_data['Outcome']

# Convert to numpy arrays
features = features.to_numpy()
target = target.to_numpy()

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=2)

# Initialize scalers and classifiers
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_results = {}

# **1. KNN Classifier**
knn_classifier = KNeighborsClassifier(p=2)
knn_classifier.fit(X_train, Y_train)
y_pred_knn = knn_classifier.predict(X_test)
model_results['KNN'] = {
    "accuracy": accuracy_score(Y_test, y_pred_knn) * 100,
    "precision": precision_score(Y_test, y_pred_knn) * 100,
    "recall": recall_score(Y_test, y_pred_knn) * 100
}

# **2. Logistic Regression**
logistic_classifier = LogisticRegression(max_iter=500)
logistic_classifier.fit(X_train, Y_train)
y_pred_logistic = logistic_classifier.predict(X_test)
model_results['Logistic Regression'] = {
    "accuracy": accuracy_score(Y_test, y_pred_logistic) * 100,
    "precision": precision_score(Y_test, y_pred_logistic) * 100,
    "recall": recall_score(Y_test, y_pred_logistic) * 100
}

# **3. Random Forest**
random_forest_classifier = RandomForestClassifier(n_estimators=100)
random_forest_classifier.fit(X_train, Y_train)
y_pred_rf = random_forest_classifier.predict(X_test)
model_results['Random Forest'] = {
    "accuracy": accuracy_score(Y_test, y_pred_rf) * 100,
    "precision": precision_score(Y_test, y_pred_rf) * 100,
    "recall": recall_score(Y_test, y_pred_rf) * 100
}

models = list(model_results.keys())
accuracies = [result['accuracy'] for result in model_results.values()]

# Plotting
plt.bar(models, accuracies, color='skyblue')
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center')

    # Save the plot

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(BASE_DIR)
plot_path = os.path.join(static_dir, 'diabetes_model_accuarcy.jpg')
plt.savefig(plot_path)
plt.close()

class_counts = diabetes_data['Outcome'].value_counts()

# Create the bar plot
plt.figure(figsize=(8, 5))
plt.bar(class_counts.index, class_counts.values, color=['skyblue', 'salmon'])
plt.xticks([0, 1], ['Non-Diabetic (0)', 'Diabetic (1)'])
plt.ylabel('Count')
plt.xlabel('Classification')
plt.title('Classification Distribution in the Dataset')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plot_path = os.path.join(static_dir, 'diabetes_outcome_count.jpg')
plt.savefig(plot_path)
plt.close()

# Save dataset and model results
with open('diabetes_data.pkl', 'wb') as file:
    pickle.dump({
        "dataset_info": dataset_info,
        "model_results": model_results,
        "scaler": scaler,
        "best_model": random_forest_classifier  # Assume Random Forest is the best model here
    }, file)

print("Dataset and model details saved to pickle!")
