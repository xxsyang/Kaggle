import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data, target_column, is_train=True):
    data = data.dropna()

    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    data['fnlwgt_large'] = data['fnlwgt'].apply(lambda x: 1 if x > 50000 else 0)

    if is_train:
        data[target_column] = data[target_column].apply(lambda x: 1 if x == '>50K' else 0)

    return data, label_encoders

def randomForest():
    try:
        train_data = pd.read_csv('data/train_final.csv')
        test_data = pd.read_csv('data/test_final.csv')
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    target_column = 'fnlwgt_large'

    train_data, train_label_encoders = preprocess_data(train_data, target_column, is_train=True)

    test_data, test_label_encoders = preprocess_data(test_data, target_column, is_train=False)

    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    clf = RandomForestClassifier(random_state=1)
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Best parameters from GridSearchCV
    best_params = grid_search.best_params_
    print("Best parameters found: ", best_params)

    # Train the model with the best parameters
    clf = RandomForestClassifier(**best_params, random_state=1)
    clf.fit(X_train, y_train)

    # Predict the response for validation dataset
    y_val_pred = clf.predict(X_val)

    # Model Accuracy, how often is the classifier correct?
    print("Validation Accuracy:", metrics.accuracy_score(y_val, y_val_pred))

    # Feature Importance
    feature_imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(feature_imp)

    # Predict the probabilities for the test dataset
    X_test = test_data.drop('ID', axis=1)  # Assuming 'ID' is not a feature
    y_test_pred_proba = clf.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'Prediction': y_test_pred_proba
    })

    submission.to_csv('submission.csv', index=False)
    print("Submission file created successfully.")

if __name__ == "__main__":
    randomForest()