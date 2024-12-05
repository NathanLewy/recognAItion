import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, Dropout
from tensorflow.keras.utils import to_categorical

# Paths for training and evaluation folders
train_folder = "/home/pierres/PROJET S7/recognAItion/data/training"
eval_folder = "/home/pierres/PROJET S7/recognAItion/data/evaluation"

# Function to load and combine data from a folder
def load_data(folder):
    combined_data = pd.DataFrame()
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    for csv_file in csv_files:
        data = pd.read_csv(os.path.join(folder, csv_file))
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data

# Load training and evaluation data
print("Loading training data...")
train_data = load_data(train_folder)

print("Loading evaluation data...")
eval_data = load_data(eval_folder)

# Separate features and labels
X_train = train_data.drop(columns=["emotion"]).values
y_train = train_data["emotion"].values
X_eval = eval_data.drop(columns=["emotion"]).values
y_eval = eval_data["emotion"].values

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)

# Reshape for CNN input (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_eval = X_eval.reshape(X_eval.shape[0], X_eval.shape[1], 1)

# Convert labels to one-hot encoding
y_train_categorical = to_categorical(y_train, num_classes=2)
y_eval_categorical = to_categorical(y_eval, num_classes=2)

# Build the CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 output classes for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the CNN model...")
history = model.fit(X_train, y_train_categorical, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
print("\nEvaluating the CNN model...")
eval_loss, eval_accuracy = model.evaluate(X_eval, y_eval_categorical, verbose=0)
print(f"Evaluation Accuracy: {eval_accuracy:.2f}")

# Make predictions
y_pred = model.predict(X_eval)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_eval, y_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_eval, y_pred_classes))
