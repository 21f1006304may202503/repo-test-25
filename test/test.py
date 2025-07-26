import os
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# set up the paths
current_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.abspath(os.path.join(current_dir,"..","train"))
model_dir = os.path.abspath(os.path.join(current_dir,".."))
model_path = os.path.join(model_dir,"model.joblib")
data_path = os.path.join(train_dir,"iris.csv")
print("model path",model_path)
print("data path",data_path)

# Load the model
model = joblib.load(model_path)

# Load the dataset
data = pd.read_csv(data_path)

# Fixtures to prepare data
@pytest.fixture
def test_data():
    """Prepare test data for evaluation."""
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test['species']
    return X_test, y_test

# Evaluation Test
def test_model_accuracy(test_data):
    """Test if model accuracy is above 80%."""
    X_test, y_test = test_data
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    assert acc >= 0.8, f"Model accuracy is too low: {acc}"

# Data Validation Test 1: Check for missing values
def test_no_missing_values():
    """Test to ensure there are no missing values in the dataset."""
    assert data.isnull().sum().sum() == 0, "Dataset contains missing values."

# Data Validation Test 2: Check feature ranges (basic sanity check)
def test_feature_ranges():
    """Test to check that feature values fall within expected ranges."""
    assert data['sepal_length'].between(4, 8).all(), "sepal_length out of expected range."
    assert data['sepal_width'].between(2, 5).all(), "sepal_width out of expected range."
    assert data['petal_length'].between(1, 7).all(), "petal_length out of expected range."
    assert data['petal_width'].between(0, 3).all(), "petal_width out of expected range."

# Data Validation Test 3: Check for expected classes
def test_expected_classes():
    """Test to check all target classes are present."""
    expected_classes = {"setosa", "versicolor", "virginica"}
    actual_classes = set(data['species'].unique())
    assert expected_classes.issubset(actual_classes), f"Missing expected classes: {expected_classes - actual_classes}"