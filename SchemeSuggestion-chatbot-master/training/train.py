import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import joblib

# Load the existing dataset and model
df = pd.read_csv("./training/dataset.csv")

X = df[['age', 'religion', 'community', 'income', 'gender', 'segment']]
y = df[['scheme', 'link']]

X_encoded = pd.get_dummies(X, columns=['religion', 'community', 'gender', 'segment'])
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Retrain the model
classifier = MultiOutputClassifier(RandomForestClassifier())
classifier.fit(X_train, y_train)

# Save the new model, replacing the existing one
joblib.dump(classifier, "./model/model.joblib", compress=True)
