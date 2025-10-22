import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Emotions mapping (focus on 4 for your model)
emotions_map = {
    '01': 'neutral',
    '04': 'sad',
    '05': 'angry',
    '03': 'happy'
}
selected_emotions = list(emotions_map.values())

# Feature Extraction Function
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        features = np.hstack([mfcc, chroma, mel])
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Load and Preprocess Dataset
def load_ravdess_data(data_path):
    features = []
    labels = []
    for actor_folder in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        for file_name in os.listdir(actor_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(actor_path, file_name)
                parts = file_name.split('-')
                if len(parts) < 3:
                    continue
                emotion_code = parts[2]
                if emotion_code in emotions_map:
                    emotion = emotions_map[emotion_code]
                    feat = extract_features(file_path)
                    if feat is not None:
                        features.append(feat)
                        labels.append(emotion)
    # Convert to numpy arrays
    X = np.array(features)
    y_str = np.array(labels)
    le = LabelEncoder()
    le.fit(selected_emotions)
    y = le.transform(y_str)
    return X, y

# Split into train/test
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

# ---------- MAIN SCRIPT ----------

# Set your data_path to the folder containing all the Actor_* folders
import os

# The path to your main dataset folder
data_path = r'D:\VocalMood - Aiml Project\ravdess'

print(f"Checking for actor folders inside: {data_path}\n")

# --- Error Handling ---
# First, check if the main ravdess directory even exists
if not os.path.isdir(data_path):
    print(f"ERROR: The directory was not found at the specified path.")
    print("Please make sure the path is correct and the dataset is downloaded.")
else:
    print("--- Found Actor Folders ---")
    # --- Loop and Print ---
    # List all items in the directory
    for actor_folder_name in os.listdir(data_path):
        # Create the full path to the item
        full_path = os.path.join(data_path, actor_folder_name)
        
        # Check if the item is a directory (e.g., "Actor_01", "Actor_02")
        # and not a file.
        if os.path.isdir(full_path):
            print(full_path)
    print("\n--------------------------")
    
# Load features and labels
X, y = load_ravdess_data(data_path)

# FIX: Reshape to (num_samples, 180, 1) for RNN input
X = X.reshape(X.shape[0], 180, 1)
print("X shape after reshape:", X.shape)  # Should be (num_samples, 180, 1)

# Train/test split
X_train, X_test, y_train, y_test = split_data(X, y)

# Optionally save features/labels for later use
np.save('X_features.npy', X)
np.save('y_labels.npy', y)

print(f"Dataset prepared! Train samples: {X_train.shape[0]}, Test samples: {X_test.shape}")
