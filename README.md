# VocalMood - Psychiatrist Edition

Advanced AI-powered voice emotion detection system designed to assist psychiatrists in analyzing patient mental health and depression indicators through vocal patterns.

## Features

- **Real-time Voice Analysis**: Live emotion detection during therapy sessions
- **Depression Risk Assessment**: AI-powered depression scoring based on vocal patterns
- **Patient Management**: Track patient progress over time with detailed records
- **Clinical Recommendations**: Automated clinical suggestions based on analysis results
- **Enhanced Accuracy**: Advanced deep learning model with attention mechanism
- **Professional Interface**: Clean, medical-grade user interface

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python complete_train.py
```

### 3. Run the Application
```bash
python integrated_app.py
```

### 4. Access the System
Open your browser and go to: `http://localhost:5000`

## System Components

### Core Files
- `integrated_app.py` - Main Flask application with all functionality
- `complete_train.py` - Enhanced model training script
- `psychiatrist_dashboard.html` - Main dashboard interface
- `analyze.html` - Patient analysis form
- `analysis_result.html` - Comprehensive results display
- `patient_history.html` - Patient records and statistics

### Features
- **File Upload Analysis**: Analyze pre-recorded patient sessions
- **Real-time Monitoring**: Live voice analysis during sessions
- **Depression Scoring**: Calculate depression risk percentages
- **Patient Tracking**: Maintain detailed patient records
- **Clinical Reports**: Generate professional analysis reports

## Clinical Use

### Depression Risk Levels
- **Low Risk (0-30%)**: Continue standard care protocol
- **Medium Risk (30-60%)**: Increase monitoring, consider therapy intensification
- **High Risk (60%+)**: Immediate psychiatric evaluation recommended

### Supported Audio Formats
- MP3, WAV, M4A files
- Recommended: 3-second minimum duration
- Sample rate: 22050 Hz (auto-converted)

## Database Setup

The system uses MySQL for patient records:

1. Install MySQL
2. Create database: `emotions_db`
3. Update connection string in `integrated_app.py` if needed

## Model Architecture

- **Enhanced CNN-LSTM**: Multi-scale feature extraction
- **Attention Mechanism**: Focus on relevant audio segments  
- **Bidirectional Processing**: Capture temporal dependencies
- **Depression Mapping**: Emotion-to-depression risk correlation

## Training Data

Uses RAVDESS dataset with 4 emotion classes:
- Happy
- Sad  
- Neutral
- Angry

Place dataset in `ravdess/` folder before training.

## Professional Use

This system is designed for licensed mental health professionals. Results should be used as supplementary information alongside clinical judgment and established diagnostic procedures.

## Support

For technical support or clinical implementation questions, refer to the documentation or contact the development team.