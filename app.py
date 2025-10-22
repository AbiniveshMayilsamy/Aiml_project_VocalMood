from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # For loading trained model
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = 'your_secret_key'


# MySQL Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost:3306/emotions_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Uploads folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Database Model
class EmotionRecord(db.Model):
    __tablename__ = 'emotion_records'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    emotion = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)


# Emotions list
emotions = ['happy', 'sad', 'neutral', 'angry']
label_encoder = LabelEncoder()
label_encoder.fit(emotions)


# Load the TRAINED model (must run train_model.py first)
model = load_model('high_accuracy_rnn_model.h5')


# Feature Extraction
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        features = np.hstack([mfcc, chroma, mel])
        return features.reshape(1, 180, 1)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


# Predict emotion
def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return 'unknown', 0.0
    
    prediction = model.predict(features)[0]
    emotion_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    emotion_array = label_encoder.inverse_transform([emotion_idx])  # Returns array like ['happy']
    emotion = emotion_array[0]  # Extract the string from the array
    print(f"Predicted: {emotion} with {confidence:.2f}% confidence")  # Debug print
    return emotion, confidence


# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and (file.filename.lower().endswith(('.mp3', '.m4a'))):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            try:
                file.save(file_path)
            except Exception as e:
                flash(f'Error saving file: {str(e)}')
                return redirect(request.url)
            
            # Predict emotion
            emotion, confidence = predict_emotion(file_path)
            
            # Store in database
            record = EmotionRecord(
                filename=filename,
                emotion=emotion,
                confidence=confidence,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            db.session.add(record)
            db.session.commit()
            
            # Safely remove the file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f"File {file_path} not found, skipping removal")
            
            return render_template('result.html', emotion=emotion, confidence=confidence)
        else:
            flash('Please upload an MP3 or M4A file')
    
    return render_template('analyze.html')


@app.route('/history')
def history():
    records = EmotionRecord.query.order_by(EmotionRecord.id.desc()).limit(20).all()
    return render_template('history.html', records=records)


if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("Database setup completed successfully!")
        except Exception as e:
            print(f"Database setup error: {e}")
    
    app.run(debug=True)