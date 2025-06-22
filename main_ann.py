# main_ann.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os

# PARAMETERS
DATASET_PATH = 'completion/train.csv'    # or 'train.csv' if you want to use your upload
TEST_PATH = 'completion/test.csv'        # or any test file you want
MODEL_SAVE_PATH = 'saved_model/ann_model'

# CREATE FOLDER TO SAVE MODEL
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# LOAD DATA
df = pd.read_csv(DATASET_PATH)
test_df = pd.read_csv(TEST_PATH)

# PREVIEW DATA
print("Data preview:\n", df.head())

# FEATURES & TARGET
features = ['mass A', 'mass B', 'mass C', 'mass D']
target = 'outcome'

# CHECK COLUMNS
print("\nColumns in dataset:", df.columns.tolist())

# EXTRACT X & y
X = df[features].values
y = df[target].values

# STANDARDIZE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SPLIT DATA
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# BUILD ANN MODEL
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# TRAIN MODEL
history = model.fit(X_train, y_train, epochs=50, batch_size=16,
                    validation_data=(X_val, y_val))

# EVALUATION
y_val_pred = (model.predict(X_val) > 0.5).astype("int32")

print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# SAVE MODEL
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")

# PREDICT ON TEST DATA
X_test_scaled = scaler.transform(test_df[features])
test_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
test_df['predicted_outcome'] = test_pred

test_df.to_csv('completion/test_with_predictions.csv', index=False)
print("\nPredictions saved to completion/test_with_predictions.csv")

# PLOT TRAINING HISTORY
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')
plt.grid()
plt.show()
