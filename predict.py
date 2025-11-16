"""
Custom Inference Script for Customer Churn Prediction
Loads saved model and makes predictions on custom input.
"""

import numpy as np
import keras
import joblib
import os
from sklearn.preprocessing import OneHotEncoder


def load_artifacts(models_dir='models'):
    """Load model, scaler, and label encoders."""
    try:
        model = keras.models.load_model(os.path.join(models_dir, 'churn_model.keras'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        label_encoders = joblib.load(os.path.join(models_dir, 'label_encoders.pkl'))
        return model, scaler, label_encoders
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model artifacts not found. Please run main.py first to train and save the model. Error: {e}")


def preprocess_input(credit_score, geography, gender, age, tenure, balance,
                     num_of_products, has_cr_card, is_active_member, estimated_salary,
                     label_encoders):
    """
    Preprocess custom input to match training data format.
    
    Args (in order):
        credit_score: int - Credit score (typically 300-850)
        geography: str - 'France', 'Spain', or 'Germany'
        gender: str - 'Male' or 'Female'
        age: int - Customer age
        tenure: int - Years with bank
        balance: float - Account balance
        num_of_products: int - Number of bank products
        has_cr_card: int - 1 if has credit card, 0 otherwise
        is_active_member: int - 1 if active, 0 otherwise
        estimated_salary: float - Estimated salary
        label_encoders: dict - Loaded label encoders
        
    Returns:
        numpy array: Preprocessed feature array
    """
    # Validate geography
    valid_geographies = ['France', 'Spain', 'Germany']
    if geography not in valid_geographies:
        raise ValueError(f"Geography must be one of {valid_geographies}, got: {geography}")
    
    # Validate gender
    valid_genders = ['Male', 'Female']
    if gender not in valid_genders:
        raise ValueError(f"Gender must be one of {valid_genders}, got: {gender}")
    
    # Encode Geography
    geography_encoder = label_encoders['geography']
    geography_encoded = geography_encoder.transform([geography])[0]
    
    # Encode Gender
    gender_encoder = label_encoders['gender']
    gender_encoded = gender_encoder.transform([gender])[0]
    
    # One-hot encode Geography (drop first to avoid dummy trap)
    # Geography order: France=0, Spain=1, Germany=2
    # After one-hot with drop='first': France=[0,0], Spain=[1,0], Germany=[0,1]
    if geography_encoded == 0:  # France
        geo_onehot = [0.0, 0]
    elif geography_encoded == 1:  # Spain
        geo_onehot = [1.0, 0]
    else:  # Germany
        geo_onehot = [0.0, 1]
    
    # Build feature array: [CreditScore, Geo1, Geo2, Gender, Age, Tenure, Balance, 
    #                       NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
    features = np.array([[
        credit_score,      # 0
        geo_onehot[0],     # 1
        geo_onehot[1],     # 2
        gender_encoded,    # 3
        age,               # 4
        tenure,            # 5
        balance,           # 6
        num_of_products,   # 7
        has_cr_card,       # 8
        is_active_member,  # 9
        estimated_salary   # 10
    ]])
    
    return features


def predict_churn(credit_score, geography, gender, age, tenure, balance,
                  num_of_products, has_cr_card, is_active_member, estimated_salary,
                  models_dir='models', threshold=0.5):
    """
    Predict customer churn for custom input.
    
    Args (in order):
        credit_score: int - Credit score
        geography: str - 'France', 'Spain', or 'Germany'
        gender: str - 'Male' or 'Female'
        age: int - Customer age
        tenure: int - Years with bank
        balance: float - Account balance
        num_of_products: int - Number of products
        has_cr_card: int - 1 if has credit card, 0 otherwise
        is_active_member: int - 1 if active, 0 otherwise
        estimated_salary: float - Estimated salary
        models_dir: str - Directory containing saved models
        threshold: float - Probability threshold for churn (default 0.5)
        
    Returns:
        dict: Prediction results with probability and binary prediction
    """
    # Load artifacts
    model, scaler, label_encoders = load_artifacts(models_dir)
    
    # Preprocess input
    features = preprocess_input(
        credit_score, geography, gender, age, tenure, balance,
        num_of_products, has_cr_card, is_active_member, estimated_salary,
        label_encoders
    )
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    probability = model.predict(features_scaled, verbose=0)[0][0]
    will_churn = probability > threshold
    
    return {
        'probability': float(probability),
        'will_churn': bool(will_churn),
        'confidence': float(abs(probability - 0.5) * 2)  # Confidence score 0-1
    }


def main():
    """Example usage with command-line input."""
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - INFERENCE")
    print("=" * 60)
    print()
    
    try:
        # Example prediction
        result = predict_churn(
            credit_score=600,
            geography='France',
            gender='Male',
            age=40,
            tenure=3,
            balance=60000,
            num_of_products=2,
            has_cr_card=1,
            is_active_member=1,
            estimated_salary=50000
        )
        
        print("Prediction Results:")
        print(f"  Churn Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
        print(f"  Will Churn: {'Yes' if result['will_churn'] else 'No'}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print()
        
        # Interactive example
        print("=" * 60)
        print("Interactive Prediction")
        print("=" * 60)
        print("Enter customer details (or press Enter for defaults):")
        print("Order: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary")
        print()
        
        credit_score = int(input("Credit Score [600]: ").strip() or "600")
        geography = input("Geography (France/Spain/Germany) [France]: ").strip() or "France"
        gender = input("Gender (Male/Female) [Male]: ").strip() or "Male"
        age = int(input("Age [40]: ").strip() or "40")
        tenure = int(input("Tenure (years) [3]: ").strip() or "3")
        balance = float(input("Balance [60000]: ").strip() or "60000")
        num_of_products = int(input("Number of Products [2]: ").strip() or "2")
        has_cr_card = int(input("Has Credit Card (1/0) [1]: ").strip() or "1")
        is_active_member = int(input("Is Active Member (1/0) [1]: ").strip() or "1")
        estimated_salary = float(input("Estimated Salary [50000]: ").strip() or "50000")
        
        result = predict_churn(
            credit_score=credit_score,
            geography=geography,
            gender=gender,
            age=age,
            tenure=tenure,
            balance=balance,
            num_of_products=num_of_products,
            has_cr_card=has_cr_card,
            is_active_member=is_active_member,
            estimated_salary=estimated_salary
        )
        
        print()
        print("=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Churn Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
        print(f"Will Churn: {'Yes' if result['will_churn'] else 'No'}")
        print(f"Confidence: {result['confidence']:.2%}")
        print()
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease run main.py first to train and save the model.")
    except ValueError as e:
        print(f"ERROR: Invalid input - {e}")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()

