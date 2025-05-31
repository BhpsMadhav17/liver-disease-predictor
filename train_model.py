import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Create synthetic liver disease dataset
def create_dataset():
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'gender': np.random.choice([0, 1], n_samples),  # 0=Female, 1=Male
        'total_bilirubin': np.random.uniform(0.3, 3.0, n_samples),
        'alkaline_phosphotase': np.random.uniform(44, 147, n_samples),
        'alamine_aminotransferase': np.random.uniform(10, 56, n_samples),
        'aspartate_aminotransferase': np.random.uniform(10, 40, n_samples),
        'total_proteins': np.random.uniform(6.0, 8.3, n_samples),
        'albumin': np.random.uniform(3.5, 5.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (liver disease) based on realistic medical logic
    disease_prob = (
        (df['total_bilirubin'] > 1.2).astype(int) * 0.3 +
        (df['alkaline_phosphotase'] > 120).astype(int) * 0.25 +
        (df['alamine_aminotransferase'] > 40).astype(int) * 0.25 +
        (df['aspartate_aminotransferase'] > 35).astype(int) * 0.2 +
        (df['age'] > 50).astype(int) * 0.15 +
        np.random.uniform(0, 0.2, n_samples)  # Add some randomness
    )
    
    df['disease'] = (disease_prob > 0.5).astype(int)
    
    return df

# Train the model
def train_model():
    print("Creating dataset...")
    df = create_dataset()
    
    # Prepare features and target
    features = ['age', 'gender', 'total_bilirubin', 'alkaline_phosphotase',
                'alamine_aminotransferase', 'aspartate_aminotransferase',
                'total_proteins', 'albumin']
    
    X = df[features]
    y = df['disease']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model saved as 'model.pkl'")
    print("Scaler saved as 'scaler.pkl'")
    
    return model, scaler, accuracy

if __name__ == "__main__":
    model, scaler, accuracy = train_model()