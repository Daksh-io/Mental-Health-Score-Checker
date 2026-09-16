import sys
import sklearn
import joblib

print("Python:", sys.executable)
print("Scikit-learn:", sklearn.__version__)
print("Scikit-learn path:", sklearn.__file__)

model = joblib.load("Mental_Health_Model.pkl")

print("Model loaded successfully!")
print(type(model))
