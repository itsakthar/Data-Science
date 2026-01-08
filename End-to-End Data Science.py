"""
CODTECH Data Science Internship
Task 3: End-to-End Data Science Project

Project: Iris Flower Classification
Tools: Pandas, Scikit-learn, Flask
Author: Shamim Akthar
"""

# -----------------------------
# Import libraries
# -----------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from flask import Flask, request, jsonify

# -----------------------------
# Step 1: Load dataset
# -----------------------------
# Using Iris dataset from sklearn
from sklearn.datasets import load_iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# -----------------------------
# Step 2: Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 3: Preprocessing + Model
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")
print("Test Accuracy:", model.score(X_test_scaled, y_test))

# -----------------------------
# Step 4: Save model + scaler
# -----------------------------
joblib.dump(model, "iris_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# Step 5: Deploy with Flask
# -----------------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']  # expecting a list of 4 values
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("iris_model.pkl")
    data_scaled = scaler.transform([data])
    prediction = model.predict(data_scaled)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)

"""
Example API Call(Bash):
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"features":[5.1, 3.5, 1.4, 0.2]}'

# Output:
{"prediction": 0}
"""
