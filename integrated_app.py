from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from werkzeug.utils import secure_filename
import threading
import time
import base64
import wave
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost:3306/emotions_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Upload Configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database Models
class PatientRecord(db.Model):
    __tablename__ = 'patient_records'
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(255), nullable=False)
    session_id = db.Column(db.String(100), nullable=False)
    emotion = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    depression_score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)

# Load Model
try:
    model = tf.keras.models.load_model('enhanced_emotion_model.h5')
    print("Enhanced model loaded")
except:
    try:
        model = tf.keras.models.load_model('high_accuracy_rnn_model.h5')
        print("Default model loaded")
    except:
        print("No model found")
        model = None

emotions = ['happy', 'sad', 'neutral', 'angry']
label_encoder = LabelEncoder()
label_encoder.fit(emotions)

# Real-time analyzer
class SimpleRealTimeAnalyzer:
    def __init__(self):
        self.is_recording = False
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.emotion_history = []
        
    def analyze_emotion(self, audio_data):
        try:
            features = self.extract_features_from_array(audio_data)
            if features is None:
                return "neutral", 0.0
            
            features = features.reshape(1, 180, 1)
            prediction = model.predict(features, verbose=0)[0]
            emotion_idx = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            emotion = emotions[emotion_idx]
            
            return emotion, confidence
        except:
            return "neutral", 0.0
    
    def extract_features_from_array(self, audio_data, sr=22050):
        try:
            mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40).T, axis=0)
            stft = np.abs(librosa.stft(audio_data))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sr).T, axis=0)
            features = np.hstack([mfcc, chroma, mel])
            
            if len(features) < 180:
                features = np.pad(features, (0, 180 - len(features)), mode='constant')
            else:
                features = features[:180]
                
            return features
        except:
            return np.zeros(180)

analyzer = SimpleRealTimeAnalyzer()

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        features = np.hstack([mfcc, chroma, mel])
        
        if len(features) < 180:
            features = np.pad(features, (0, 180 - len(features)), mode='constant')
        else:
            features = features[:180]
            
        return features.reshape(1, 180, 1)
    except:
        return None

def calculate_depression_score(emotion, confidence):
    """Calculate depression risk score based on emotion analysis"""
    depression_weights = {
        'sad': 0.8,
        'angry': 0.6,
        'neutral': 0.3,
        'happy': 0.1
    }
    base_score = depression_weights.get(emotion, 0.3)
    return min(base_score * (confidence / 100) * 100, 100)

def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return 'neutral', 0.0, 0.0
    
    prediction = model.predict(features)[0]
    emotion_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    emotion = emotions[emotion_idx]
    depression_score = calculate_depression_score(emotion, confidence)
    
    return emotion, confidence, depression_score

@app.route('/')
def index():
    return render_template('psychiatrist_dashboard.html')

@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        patient_name = request.form.get('patient_name', 'Anonymous')
        session_id = request.form.get('session_id', f'session_{int(time.time())}')
        
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and file.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            emotion, confidence, depression_score = predict_emotion(file_path)
            
            # Save to database
            record = PatientRecord(
                patient_name=patient_name,
                session_id=session_id,
                emotion=emotion,
                confidence=confidence,
                depression_score=depression_score,
                notes=request.form.get('notes', '')
            )
            db.session.add(record)
            db.session.commit()
            
            os.remove(file_path)
            
            return render_template('analysis_result.html', 
                                 emotion=emotion, 
                                 confidence=confidence,
                                 depression_score=depression_score,
                                 patient_name=patient_name,
                                 session_id=session_id,
                                 datetime=datetime)
        else:
            flash('Please upload an audio file (MP3, WAV, M4A)')
    
    return render_template('analyze.html')

@app.route('/history')
def history():
    records = PatientRecord.query.order_by(PatientRecord.timestamp.desc()).limit(50).all()
    return render_template('patient_history.html', records=records)

@app.route('/patient/<patient_name>')
def patient_detail(patient_name):
    records = PatientRecord.query.filter_by(patient_name=patient_name).order_by(PatientRecord.timestamp.desc()).all()
    return render_template('patient_detail.html', records=records, patient_name=patient_name)

@socketio.on('analyze_audio_chunk')
def analyze_audio_chunk(data):
    try:
        audio_data = base64.b64decode(data['audio'])
        
        # Convert to numpy array (simplified)
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        
        emotion, confidence = analyzer.analyze_emotion(audio_array)
        depression_score = calculate_depression_score(emotion, confidence)
        
        emit('emotion_result', {
            'emotion': emotion,
            'confidence': round(confidence, 1),
            'depression_score': round(depression_score, 1),
            'timestamp': time.time()
        })
        
    except Exception as e:
        emit('error', {'message': f'Analysis failed: {str(e)}'})

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("Database setup completed")
        except Exception as e:
            print(f"Database setup error: {e}")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)