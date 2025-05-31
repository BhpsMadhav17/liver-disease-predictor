from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import sqlite3
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize database
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  age INTEGER,
                  gender TEXT,
                  total_bilirubin REAL,
                  alkaline_phosphotase REAL,
                  alamine_aminotransferase REAL,
                  aspartate_aminotransferase REAL,
                  total_proteins REAL,
                  albumin REAL,
                  prediction TEXT,
                  probability REAL,
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = int(request.form['age'])
        gender = 1 if request.form['gender'] == 'Male' else 0
        total_bilirubin = float(request.form['total_bilirubin'])
        alkaline_phosphotase = float(request.form['alkaline_phosphotase'])
        alamine_aminotransferase = float(request.form['alamine_aminotransferase'])
        aspartate_aminotransferase = float(request.form['aspartate_aminotransferase'])
        total_proteins = float(request.form['total_proteins'])
        albumin = float(request.form['albumin'])
        
        # Create feature array
        features = np.array([[age, gender, total_bilirubin, alkaline_phosphotase, 
                            alamine_aminotransferase, aspartate_aminotransferase, 
                            total_proteins, albumin]])
        
        # Scale features (important!)
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0].max()
        
        # Store in database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('''INSERT INTO predictions 
                     (age, gender, total_bilirubin, alkaline_phosphotase, 
                      alamine_aminotransferase, aspartate_aminotransferase, 
                      total_proteins, albumin, prediction, probability, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (age, 'Male' if gender == 1 else 'Female', total_bilirubin, 
                   alkaline_phosphotase, alamine_aminotransferase, 
                   aspartate_aminotransferase, total_proteins, albumin,
                   'Disease' if prediction == 1 else 'No Disease', 
                   probability, datetime.now()))
        conn.commit()
        conn.close()
        
        result = 'Disease Detected' if prediction == 1 else 'No Disease'
        confidence = round(probability * 100, 2)
        
        return render_template('result.html', 
                             prediction=result,
                             confidence=confidence)
    
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/analytics')
def analytics():
    conn = sqlite3.connect('database.db')
    try:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)
        if len(df) == 0:
            stats = {'total_predictions': 0, 'disease_cases': 0, 'avg_age': 0}
        else:
            stats = {
                'total_predictions': len(df),
                'disease_cases': len(df[df['prediction'] == 'Disease']),
                'avg_age': round(df['age'].mean(), 1)
            }
    except:
        stats = {'total_predictions': 0, 'disease_cases': 0, 'avg_age': 0}
    finally:
        conn.close()
    
    return render_template('analytics.html', stats=stats)

@app.route('/model-info')
def model_info():
    # Your model's performance metrics
    metrics = {
        'accuracy': '92.5%',
        'algorithm': 'Random Forest',
        'features': 8,
        'samples_trained': 800
    }
    return render_template('model_info.html', metrics=metrics)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)