import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load training and evaluation data
training_file = "/home/pierres/PROJET S7/recognAItion/data/training_combined.csv"
evaluation_file = "/home/pierres/PROJET S7/recognAItion/data/evaluation_combined.csv"

# Load datasets
train_data = pd.read_csv(training_file)
eval_data = pd.read_csv(evaluation_file)

# Separate features and target for training data
X_train = train_data.drop(columns=["emotion"])  # Features
y_train = train_data["emotion"]  # Target

# Separate features and target for evaluation data
X_eval = eval_data.drop(columns=["emotion"])  # Features
y_eval = eval_data["emotion"]  # Target

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)  # n_estimators defines number of trees

# Train the model
print("Training the random forest model...")
model.fit(X_train, y_train)

# Make predictions on evaluation data
print("Evaluating the model...")
y_pred = model.predict(X_eval)

# Calculate accuracy
accuracy = accuracy_score(y_eval, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_eval, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_eval, y_pred))
