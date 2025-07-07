
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset (use the Kaggle credit card dataset)
df = pd.read_csv("creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fraud_model.pkl")
print("✅ Model trained and saved.")
