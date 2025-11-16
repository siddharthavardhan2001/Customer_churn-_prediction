"""
Customer Churn Prediction using Artificial Neural Network (ANN)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib
import os

# Handle KerasClassifier import for different Keras versions
try:
    from keras.wrappers.scikit_learn import KerasClassifier
except ImportError:
    try:
        # Try TensorFlow Keras wrapper
        from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
    except ImportError:
        # Keras 3.x doesn't have scikit-learn wrapper by default
        # We'll create a simple wrapper or skip those features
        KerasClassifier = None
        print("Warning: KerasClassifier not available. Cross-validation and GridSearch will be skipped.")

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def load_data(csv_path='Churn_Modelling.csv'):
    """
    Load the dataset and extract features and target variable.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    
    dataset = pd.read_csv(csv_path)
    print(f"Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    
    # Extract features (columns 3 to 12) and target (column 13)
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print()
    
    return X, y


def encode_categorical_features(X):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        X (numpy array): Feature matrix
        
    Returns:
        tuple: (X_encoded, label_encoders) where label_encoders is a dict
    """
    print("=" * 60)
    print("STEP 2: Encoding Categorical Features")
    print("=" * 60)
    
    # Create copies to avoid modifying original
    X_encoded = X.copy()
    
    # Encode Geography (column 1)
    labelencoder_X_1 = LabelEncoder()
    X_encoded[:, 1] = labelencoder_X_1.fit_transform(X_encoded[:, 1])
    print(f"Geography encoded: {labelencoder_X_1.classes_}")
    
    # Encode Gender (column 2)
    labelencoder_X_2 = LabelEncoder()
    X_encoded[:, 2] = labelencoder_X_2.fit_transform(X_encoded[:, 2])
    print(f"Gender encoded: {labelencoder_X_2.classes_}")
    print()
    
    label_encoders = {
        'geography': labelencoder_X_1,
        'gender': labelencoder_X_2
    }
    
    return X_encoded, label_encoders


def apply_onehot_encoding(X):
    """
    Apply one-hot encoding to categorical features.
    Handles both old and new scikit-learn versions.
    
    Args:
        X (numpy array): Feature matrix with encoded categorical variables
        
    Returns:
        numpy array: One-hot encoded feature matrix
    """
    print("=" * 60)
    print("STEP 3: Applying One-Hot Encoding")
    print("=" * 60)
    
    # Handle one-hot encoding for Geography (column 1)
    # We need to encode only column 1 (Geography) which has 3 categories
    # After encoding, we'll have 2 columns (dropping first to avoid dummy trap)
    
    # Extract Geography column (column 1)
    geography_col = X[:, [1]]
    
    # Apply one-hot encoding with drop='first' to avoid dummy variable trap
    try:
        # Try new scikit-learn API
        onehotencoder = OneHotEncoder(drop='first', sparse_output=False)
    except TypeError:
        # Fallback for older versions
        try:
            onehotencoder = OneHotEncoder(drop='first', sparse=False)
        except TypeError:
            # Very old version - use categorical_features (deprecated)
            onehotencoder = OneHotEncoder(categorical_features=[1], sparse=False)
            X_encoded = onehotencoder.fit_transform(X)
            X_encoded = X_encoded[:, 1:]  # Remove first column
            print("One-hot encoding applied (legacy method)")
            print(f"Features shape after encoding: {X_encoded.shape}")
            print()
            return X_encoded
    
    # Encode Geography column
    geography_encoded = onehotencoder.fit_transform(geography_col)
    
    # Combine: CreditScore (col 0) + Geography encoded (2 cols) + remaining columns (2:)
    X_encoded = np.hstack([X[:, [0]], geography_encoded, X[:, 2:]])
    
    print("One-hot encoding applied")
    print(f"Features shape after encoding: {X_encoded.shape}")
    print()
    
    return X_encoded


def split_and_scale_data(X, y, test_size=0.2, random_state=0):
    """
    Split data into train/test sets and apply feature scaling.
    
    Args:
        X (numpy array): Feature matrix
        y (numpy array): Target vector
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("=" * 60)
    print("STEP 4: Train-Test Split and Feature Scaling")
    print("=" * 60)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print("Feature scaling applied (StandardScaler)")
    print()
    
    return X_train, X_test, y_train, y_test, sc


# ============================================================================
# ANN MODEL BUILDING
# ============================================================================

def build_ann_model(input_dim=11, units=6, optimizer='adam'):
    """
    Build a basic ANN model.
    
    Args:
        input_dim (int): Number of input features
        units (int): Number of units in hidden layers
        optimizer (str): Optimizer to use
        
    Returns:
        keras.Model: Compiled ANN model
    """
    classifier = Sequential()
    
    # Input layer and first hidden layer
    classifier.add(Dense(
        units=units,
        kernel_initializer='uniform',
        activation='relu',
        input_dim=input_dim
    ))
    
    # Second hidden layer
    classifier.add(Dense(
        units=units,
        kernel_initializer='uniform',
        activation='relu'
    ))
    
    # Output layer
    classifier.add(Dense(
        units=1,
        kernel_initializer='uniform',
        activation='sigmoid'
    ))
    
    # Compile the ANN
    classifier.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return classifier


def build_ann_model_with_dropout(input_dim=11, units=6, dropout_rate=0.1, optimizer='adam'):
    """
    Build an ANN model with dropout regularization.
    
    Args:
        input_dim (int): Number of input features
        units (int): Number of units in hidden layers
        dropout_rate (float): Dropout rate
        optimizer (str): Optimizer to use
        
    Returns:
        keras.Model: Compiled ANN model with dropout
    """
    classifier = Sequential()
    
    # Input layer and first hidden layer
    classifier.add(Dense(
        units=units,
        kernel_initializer='uniform',
        activation='relu',
        input_dim=input_dim
    ))
    classifier.add(Dropout(rate=dropout_rate))
    
    # Second hidden layer
    classifier.add(Dense(
        units=units,
        kernel_initializer='uniform',
        activation='relu'
    ))
    classifier.add(Dropout(rate=dropout_rate))
    
    # Output layer
    classifier.add(Dense(
        units=1,
        kernel_initializer='uniform',
        activation='sigmoid'
    ))
    
    # Compile the ANN
    classifier.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return classifier


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_basic_model(X_train, y_train, batch_size=10, epochs=100, verbose=1):
    """
    Train a basic ANN model.
    
    Args:
        X_train (numpy array): Training features
        y_train (numpy array): Training target
        batch_size (int): Batch size for training
        epochs (int): Number of epochs
        verbose (int): Verbosity level
        
    Returns:
        keras.Model: Trained model
    """
    print("=" * 60)
    print("STEP 5: Training Basic ANN Model")
    print("=" * 60)
    
    classifier = build_ann_model(input_dim=X_train.shape[1])
    print("Model architecture:")
    classifier.summary()
    print()
    
    print(f"Training with batch_size={batch_size}, epochs={epochs}")
    history = classifier.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose
    )
    
    print("\nTraining completed!")
    print()
    
    return classifier, history


def evaluate_model(classifier, X_test, y_test):
    """
    Evaluate the trained model on test set.
    
    Args:
        classifier (keras.Model): Trained model
        X_test (numpy array): Test features
        y_test (numpy array): Test target
        
    Returns:
        tuple: (accuracy, confusion_matrix)
    """
    print("=" * 60)
    print("STEP 6: Evaluating Model")
    print("=" * 60)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate accuracy
    accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    
    print("Confusion Matrix:")
    print(cm)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    return accuracy, cm


def predict_single_customer(classifier, scaler, customer_data):
    """
    Predict churn for a single customer.
    
    Args:
        classifier (keras.Model): Trained model
        scaler (StandardScaler): Fitted scaler
        customer_data (list): Customer features in order:
            [Geography_encoded, Gender_encoded, CreditScore, Age, Tenure,
             Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
            Note: Geography should be one-hot encoded (2 values for 3 countries)
    
    Returns:
        bool: True if customer will churn, False otherwise
    """
    print("=" * 60)
    print("STEP 7: Predicting for Single Customer")
    print("=" * 60)
    
    # Transform customer data
    customer_array = np.array([customer_data])
    customer_scaled = scaler.transform(customer_array)
    
    # Make prediction
    prediction = classifier.predict(customer_scaled)
    will_churn = (prediction > 0.5)[0][0]
    
    print(f"Customer data: {customer_data}")
    print(f"Churn probability: {prediction[0][0]:.4f}")
    print(f"Will churn: {will_churn}")
    print()
    
    return will_churn


# ============================================================================
# K-FOLD CROSS VALIDATION
# ============================================================================

def build_classifier_for_cv():
    """
    Build classifier function for KerasClassifier wrapper.
    Used in cross-validation.
    
    Returns:
        keras.Model: Compiled model
    """
    return build_ann_model(input_dim=11, units=6, optimizer='adam')


def perform_kfold_cross_validation(X_train, y_train, cv=10, batch_size=10, epochs=100, n_jobs=1):
    """
    Perform k-fold cross-validation on the ANN model.
    
    Args:
        X_train (numpy array): Training features
        y_train (numpy array): Training target
        cv (int): Number of folds
        batch_size (int): Batch size
        epochs (int): Number of epochs
        n_jobs (int): Number of parallel jobs
        
    Returns:
        tuple: (mean_accuracy, std_accuracy, accuracies)
    """
    print("=" * 60)
    print("STEP 8: K-Fold Cross Validation")
    print("=" * 60)
    
    if KerasClassifier is None:
        print("ERROR: KerasClassifier is not available in this Keras version.")
        print("Please install scikeras: pip install scikeras")
        print("Or use Keras 2.x for this feature.")
        print()
        return None, None, None
    
    # Handle both old and new KerasClassifier API
    try:
        classifier = KerasClassifier(
            build_fn=build_classifier_for_cv,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0
        )
    except TypeError:
        # Newer Keras versions use 'model' instead of 'build_fn'
        try:
            classifier = KerasClassifier(
                model=build_classifier_for_cv,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0
            )
        except Exception as e:
            print(f"Error creating KerasClassifier: {e}")
            print("Skipping cross-validation.")
            return None, None, None
    
    print(f"Performing {cv}-fold cross-validation...")
    accuracies = cross_val_score(
        estimator=classifier,
        X=X_train,
        y=y_train,
        cv=cv,
        n_jobs=n_jobs
    )
    
    mean_accuracy = accuracies.mean()
    std_accuracy = accuracies.std()
    
    print(f"Mean accuracy: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
    print(f"Standard deviation: {std_accuracy:.4f}")
    print()
    
    return mean_accuracy, std_accuracy, accuracies


# ============================================================================
# GRID SEARCH
# ============================================================================

def build_classifier_for_gridsearch(optimizer='adam'):
    """
    Build classifier function for GridSearchCV.
    
    Args:
        optimizer (str): Optimizer to use
        
    Returns:
        keras.Model: Compiled model
    """
    return build_ann_model(input_dim=11, units=6, optimizer=optimizer)


def perform_grid_search(X_train, y_train, parameters=None, cv=10, n_jobs=1):
    """
    Perform grid search to find best hyperparameters.
    
    Args:
        X_train (numpy array): Training features
        y_train (numpy array): Training target
        parameters (dict): Parameter grid for grid search
        cv (int): Number of folds
        n_jobs (int): Number of parallel jobs
        
    Returns:
        tuple: (best_params, best_accuracy, grid_search)
    """
    print("=" * 60)
    print("STEP 9: Grid Search for Best Parameters")
    print("=" * 60)
    
    if KerasClassifier is None:
        print("ERROR: KerasClassifier is not available in this Keras version.")
        print("Please install scikeras: pip install scikeras")
        print("Or use Keras 2.x for this feature.")
        print()
        return None, None, None
    
    if parameters is None:
        parameters = {
            'batch_size': [25, 32],
            'epochs': [100, 500],
            'optimizer': ['adam', 'rmsprop']
        }
    
    # Handle both old and new KerasClassifier API
    try:
        classifier = KerasClassifier(build_fn=build_classifier_for_gridsearch, verbose=0)
    except TypeError:
        # Newer Keras versions use 'model' instead of 'build_fn'
        try:
            classifier = KerasClassifier(model=build_classifier_for_gridsearch, verbose=0)
        except Exception as e:
            print(f"Error creating KerasClassifier: {e}")
            print("Skipping grid search.")
            return None, None, None
    
    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=parameters,
        scoring='accuracy',
        cv=cv,
        n_jobs=n_jobs
    )
    
    print("Starting grid search (this may take a long time)...")
    print(f"Parameter grid: {parameters}")
    print()
    
    grid_search = grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    
    print("Grid search completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print()
    
    return best_params, best_accuracy, grid_search


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the complete pipeline.
    """
    print("\n" + "=" * 60)
    print("CUSTOMER CHURN PREDICTION USING ANN")
    print("=" * 60 + "\n")
    
    # Set random seeds for reproducibility
    np.random.seed(0)
    import tensorflow as tf
    tf.random.set_seed(0)
    
    # ========================================================================
    # DATA PREPROCESSING
    # ========================================================================
    X, y = load_data('Churn_Modelling.csv')
    X_encoded, label_encoders = encode_categorical_features(X)
    X_onehot = apply_onehot_encoding(X_encoded)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(
        X_onehot, y, test_size=0.2, random_state=0
    )
    
    # ========================================================================
    # BASIC ANN MODEL
    # ========================================================================
    classifier, history = train_basic_model(
        X_train, y_train,
        batch_size=10,
        epochs=100,
        verbose=1
    )
    
    accuracy, cm = evaluate_model(classifier, X_test, y_test)
    
    # Predict for a single customer example
    # Example: France, Male, CreditScore=600, Age=40, Tenure=3,
    #          Balance=60000, NumOfProducts=2, HasCrCard=1, IsActiveMember=1,
    #          EstimatedSalary=50000
    # Note: Geography one-hot: France=[0,0], Spain=[1,0], Germany=[0,1]
    customer_example = [0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]
    predict_single_customer(classifier, scaler, customer_example)
    
    # ========================================================================
    # K-FOLD CROSS VALIDATION
    # ========================================================================
    # Uncomment to run (takes time)
    # mean_acc, std_acc, accuracies = perform_kfold_cross_validation(
    #     X_train, y_train, cv=10, batch_size=10, epochs=100, n_jobs=1
    # )
    
    # ========================================================================
    # ANN WITH DROPOUT
    # ========================================================================
    print("=" * 60)
    print("STEP 10: Training ANN with Dropout")
    print("=" * 60)
    classifier_dropout = build_ann_model_with_dropout(
        input_dim=X_train.shape[1],
        units=6,
        dropout_rate=0.1,
        optimizer='adam'
    )
    print("Model architecture with dropout:")
    classifier_dropout.summary()
    print()
    
    print("Training with dropout...")
    classifier_dropout.fit(
        X_train, y_train,
        batch_size=10,
        epochs=100,
        verbose=1
    )
    print("\nTraining with dropout completed!")
    print()
    
    accuracy_dropout, cm_dropout = evaluate_model(classifier_dropout, X_test, y_test)
    
    # ========================================================================
    # SAVE MODEL AND PREPROCESSING ARTIFACTS
    # ========================================================================
    print("=" * 60)
    print("STEP 11: Saving Model and Preprocessing Artifacts")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the dropout model (better performance)
    model_path = os.path.join(models_dir, 'churn_model.keras')
    classifier_dropout.save(model_path)
    print(f"Model saved: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved: {scaler_path}")
    
    # Save label encoders
    encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
    joblib.dump(label_encoders, encoders_path)
    print(f"Label encoders saved: {encoders_path}")
    
    print("\nAll artifacts saved successfully!")
    print()
    
    # ========================================================================
    # GRID SEARCH (OPTIONAL - TAKES VERY LONG TIME)
    # ========================================================================
    # Uncomment to run grid search (can take hours!)
    # best_params, best_acc, grid_search = perform_grid_search(
    #     X_train, y_train,
    #     parameters={
    #         'batch_size': [25, 32],
    #         'epochs': [100, 500],
    #         'optimizer': ['adam', 'rmsprop']
    #     },
    #     cv=10,
    #     n_jobs=1
    # )
    
    print("=" * 60)
    print("EXECUTION COMPLETED")
    print("=" * 60)
    print("\nSummary:")
    print(f"Basic Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Dropout Model Accuracy: {accuracy_dropout:.4f} ({accuracy_dropout*100:.2f}%)")
    print(f"\nModel saved to: {models_dir}/")
    print("Use predict.py for custom inference.")
    print()


if __name__ == "__main__":
    main()

