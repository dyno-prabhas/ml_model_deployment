import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load data
diabetes_data = pd.read_csv('./diabetes.csv')  # Make sure the 'diabetes.csv' file is in the same folder

# Separate features and target
features = diabetes_data.drop(columns='Outcome', axis=1).to_numpy()
target = diabetes_data['Outcome'].to_numpy()

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=2)

# Train model
classifier = KNeighborsClassifier(p=2)
classifier.fit(X_train, Y_train)

# Evaluate model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(classifier, 'diabetes_model.pkl')
print("Model saved as 'diabetes_model.pkl'")
