# train_model.py

from joblib import dump
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Create and train the LogisticRegression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model to a file
dump(model, 'iris_model.joblib')

print("Model trained and saved successfully!")
