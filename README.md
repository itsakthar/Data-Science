# Data-Science
# Task 1: Data Pipeline Development (ETL)

## Objective
Build an automated ETL pipeline using Pandas and Scikit-learn.

## Steps Implemented
- Data extraction using Pandas
- Handling missing values
- Feature scaling
- One-hot encoding
- Pipeline automation using ColumnTransformer

## Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn

## Output
Clean, transformed, ML-ready dataset.

Task 2: Deep Learning Project – MNIST Classification
Objective
Build a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset.
Tools Used
- TensorFlow / Keras
- Matplotlib
Steps
- Loaded and normalized MNIST dataset.
- Built a CNN with convolutional and pooling layers.
- Trained the model for 5 epochs with validation split.
- Evaluated test accuracy.
- Visualized results with accuracy and loss plots.
- Saved trained model as mnist_cnn_model.h5.
Results
- Final test accuracy: ~98%
- Accuracy and loss curves plotted.
- Sample predictions visualized with true vs predicted labels.

Task 3: End‑to‑End Data Science Project – Iris Classification with Flask API
Objective
Train a machine learning model on the Iris dataset and deploy it as a REST API using Flask.
Tools Used
- Pandas
- Scikit‑learn
- Flask
- Joblib
Steps
- Loaded Iris dataset.
- Preprocessed features using StandardScaler.
- Trained Logistic Regression classifier.
- Saved model and scaler (iris_model.pkl, scaler.pkl).
- Built Flask API with /predict endpoint.
Example API Call
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"features":[5.1, 3.5, 1.4, 0.2]}'


Results
- Model achieved high accuracy on test data.
- API returns predictions in JSON format, e.g.:
{"prediction": 0}



Task 4: Optimization Project – Resource Allocation with Linear Programming
Objective
Use Linear Programming to maximize profit given resource constraints.
Tools Used
- PuLP
- Matplotlib
Steps
- Defined decision variables (Product A and Product B).
- Created objective function to maximize profit.
- Added resource constraints.
- Solved optimization problem using PuLP.
- Visualized feasible region and optimal solution point.
Results
- Optimal production quantities calculated.
- Maximum profit displayed.
- Plot shows constraints, feasible region, and optimal solution point.



