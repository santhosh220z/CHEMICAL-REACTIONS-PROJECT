# main_ann_reactions.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os

# PARAMETERS
DATASET_PATH = 'reactions.csv'

# Save model to current directory with proper extension
MODEL_SAVE_PATH = 'ann_model_reactions.keras'

# LOAD DATA
df = pd.read_csv(DATASET_PATH)

# PREVIEW DATA
print("Data preview:\n", df.head())
print("\nColumns:", df.columns.tolist())

# FEATURES & TARGET
features = ['temperature', 'pH', 'concentration']
target = 'success'

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

# EARLY STOPPING
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# TRAIN MODEL
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# EVALUATION
y_val_pred = (model.predict(X_val) > 0.5).astype("int32")

print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

# SAVE MODEL
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved to {MODEL_SAVE_PATH}")

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

# PREDICT NEW INPUT
def predict_new_reaction(temp, ph, conc):
    input_scaled = scaler.transform([[temp, ph, conc]])
    prediction = model.predict(input_scaled)[0][0]
    success = 1 if prediction > 0.5 else 0
    print(f"\nPrediction → Success Probability: {prediction:.4f} → Success={success}")

# USER INPUT
try:
    temp_input = float(input("\nEnter Temperature: "))
    ph_input = float(input("Enter pH: "))
    conc_input = float(input("Enter Concentration: "))
    predict_new_reaction(temp_input, ph_input, conc_input)
except:
    print("Invalid input. Please enter numeric values.")
