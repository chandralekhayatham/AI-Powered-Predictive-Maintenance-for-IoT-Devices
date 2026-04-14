import pandas as pd

# Load data
data = pd.read_csv("data.csv")

# Show data
print("Dataset:\n", data)

# Features & Target
X = data[['temperature', 'vibration', 'current']]
y = data['failure']

# Train model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))