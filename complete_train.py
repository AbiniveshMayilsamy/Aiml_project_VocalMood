#!/usr/bin/env python3
"""
Complete Training Script for VocalMood Psychiatrist Edition
Enhanced model with depression analysis capabilities
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import librosa

def extract_enhanced_features(file_path, sr=22050):
    """Extract comprehensive audio features for emotion detection"""
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=3.0)
        
        # Pad or trim to consistent length
        if len(y) < sr * 3:
            y = np.pad(y, (0, sr * 3 - len(y)), mode='constant')
        else:
            y = y[:sr * 3]
        
        # Extract multiple features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        
        # Additional features for better accuracy
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Combine all features
        features = np.hstack([mfcc, chroma, mel, [spectral_centroid, spectral_rolloff, zero_crossing_rate]])
        
        # Ensure consistent feature size
        if len(features) < 180:
            features = np.pad(features, (0, 180 - len(features)), mode='constant')
        else:
            features = features[:180]
            
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return np.zeros(180)

def create_enhanced_model(input_shape=(180, 1), num_classes=4):
    """Create enhanced model with attention mechanism"""
    inputs = Input(shape=input_shape)
    
    # Multi-scale CNN layers
    x1 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.2)(x1)
    
    x2 = Conv1D(64, 5, activation='relu', padding='same')(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    
    # Concatenate multi-scale features
    x = Concatenate()([x1, x2])
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # Attention mechanism
    attention = Dense(1, activation='tanh')(x)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1])(attention)
    
    # Apply attention
    x = Multiply()([x, attention])
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def prepare_dataset(data_path):
    """Prepare dataset with enhanced feature extraction"""
    print("🎵 Preparing enhanced dataset for psychiatrist analysis...")
    
    emotions_map = {
        '01': 'neutral',
        '03': 'happy', 
        '04': 'sad',
        '05': 'angry'
    }
    
    features = []
    labels = []
    processed_files = 0
    
    for actor_folder in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue
            
        print(f"Processing {actor_folder}...")
        
        for file_name in os.listdir(actor_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(actor_path, file_name)
                parts = file_name.split('-')
                
                if len(parts) < 3:
                    continue
                    
                emotion_code = parts[2]
                if emotion_code in emotions_map:
                    emotion = emotions_map[emotion_code]
                    
                    feat = extract_enhanced_features(file_path)
                    if feat is not None:
                        features.append(feat)
                        labels.append(emotion)
                        processed_files += 1
                        
                        if processed_files % 100 == 0:
                            print(f"  Processed {processed_files} files...")
    
    print(f"✅ Processed {processed_files} audio files")
    
    # Convert to numpy arrays
    X = np.array(features)
    y_str = np.array(labels)
    
    # Encode labels
    le = LabelEncoder()
    le.fit(list(emotions_map.values()))
    y = le.transform(y_str)
    
    # Reshape for model input
    X = X.reshape(X.shape[0], 180, 1)
    
    print(f"📊 Dataset shape: {X.shape}")
    print(f"📊 Labels shape: {y.shape}")
    
    # Print class distribution
    print("\n📈 Class Distribution:")
    unique, counts = np.unique(y_str, return_counts=True)
    for emotion, count in zip(unique, counts):
        print(f"  {emotion}: {count} samples")
    
    return X, y, le

def train_model(X, y):
    """Train the enhanced model"""
    print(f"\n🚀 Training enhanced psychiatrist model...")
    
    # Convert labels to categorical
    y_cat = to_categorical(y, num_classes=4)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training samples: {X_train.shape[0]}")
    print(f"📊 Validation samples: {X_val.shape[0]}")
    
    # Create model
    model = create_enhanced_model()
    
    # Compile model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n🏗️  Model Architecture:")
    model.summary()
    
    # Compute class weights
    y_int = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print(f"\n🎯 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate final performance
    print(f"\n📊 Final Evaluation:")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    
    return model, history

def main():
    """Main training function"""
    print("🧠 VocalMood Psychiatrist Edition - Enhanced Training")
    print("=" * 60)
    
    # Check dataset
    data_path = r'D:\VocalMood - Aiml Project\ravdess'
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at: {data_path}")
        print("Please ensure the RAVDESS dataset is downloaded.")
        return
    
    try:
        # Prepare dataset
        X, y, label_encoder = prepare_dataset(data_path)
        
        # Train model
        model, history = train_model(X, y)
        
        # Save model
        model.save('enhanced_emotion_model.h5')
        print(f"\n✅ Enhanced model saved as 'enhanced_emotion_model.h5'")
        
        # Save training data
        np.save('enhanced_X_features.npy', X)
        np.save('enhanced_y_labels.npy', y)
        print(f"💾 Training data saved")
        
        # Test model
        print(f"\n🧪 Quick model test:")
        test_sample = X[0:1]
        prediction = model.predict(test_sample, verbose=0)
        emotions = ['happy', 'sad', 'neutral', 'angry']
        predicted_emotion = emotions[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        print(f"Test prediction: {predicted_emotion} ({confidence:.1f}% confidence)")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"✅ Your VocalMood Psychiatrist system is ready!")
        print(f"✅ Run 'python integrated_app.py' to start the application")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return

if __name__ == "__main__":
    main()