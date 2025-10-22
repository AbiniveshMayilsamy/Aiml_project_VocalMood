import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os

# --- 1. DIAGNOSE THE IMBALANCE ---
# Load labels and print the count for each class to confirm the imbalance
print("--- Class Distribution ---")
y = np.load('y_labels.npy')
print(pd.Series(y).value_counts())
print("--------------------------\n")

# Load features
X = np.load('X_features.npy')

# Compute class weights to help the model pay more attention to minority classes
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))
print("Computed Class Weights:", class_weight_dict)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Added stratify

# One-hot encode the integer labels
num_classes = len(np.unique(y)) # More robust way to define num_classes
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# --- 2. MODIFY THE MODEL DEFINITION ---
# Create RNN model with a lower learning rate
def create_rnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Use the Adam optimizer with a lower learning rate to encourage more careful learning
    optimizer = Adam(learning_rate=0.0001)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create the model instance
model = create_rnn_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
model.summary()


# --- 3. ADD EARLY STOPPING TO TRAINING ---
# Define the EarlyStopping callback
# This will monitor the validation loss and stop training if it doesn't improve for 10 epochs.
# It will also restore the model weights from the best-performing epoch.
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

# Train the model with the class weights and the early stopping callback
print("\n--- Starting Model Training ---")
history = model.fit(
    X_train,
    y_train_cat,
    epochs=100,  # High number, but EarlyStopping will find the best one
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    class_weight=class_weight_dict,
    callbacks=[early_stopping] # Pass the callback here
)

# --- 4. EVALUATE AND SAVE ---
# Evaluate the model on the test set
print("\n--- Evaluating Model ---")
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save('emotion_recognition_rnn_model.h5')
print("\nModel saved successfully as 'emotion_recognition_rnn_model.h5'")