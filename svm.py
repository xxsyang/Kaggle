import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(data, target_column=None, is_train=True):
    data = data.dropna()

    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    if is_train and target_column:
        data[target_column] = data[target_column].apply(lambda x: 1 if x == '>50K' else 0)

    data['fnlwgt_large'] = data['fnlwgt'].apply(lambda x: 1 if x > 50000 else 0)

    return data, label_encoders

def svm_model():
    try:
        train_data = pd.read_csv('data/train_final.csv')
        test_data = pd.read_csv('data/test_final.csv')
        print("Data loaded successfully.")
        print("Train data columns:", train_data.columns)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    target_column = 'fnlwgt_large'

    train_data, train_label_encoders = preprocess_data(train_data, target_column, is_train=True)

    test_data, test_label_encoders = preprocess_data(test_data, is_train=False)

    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(test_data.drop('ID', axis=1))  # Assuming 'ID' is not a feature

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }

    clf = SVC(probability=True, random_state=1)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best parameters from GridSearchCV
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    # Train the model with the best parameters
    clf = SVC(**best_params, probability=True, random_state=1)
    clf.fit(X_train, y_train)

    # Predict the response for validation dataset
    y_val_pred = clf.predict(X_val)

    # Model Accuracy, how often is the classifier correct?
    print("Validation Accuracy:", metrics.accuracy_score(y_val, y_val_pred))

    # Predict the probabilities for the test dataset
    y_test_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Create a submission DataFrame
    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'Prediction': y_test_pred_proba
    })

    # Save the submission DataFrame to a CSV file
    submission.to_csv('submission.csv', index=False)
    print("Submission file created successfully.")

# Ensure the function is called
if __name__ == "__main__":
    svm_model()