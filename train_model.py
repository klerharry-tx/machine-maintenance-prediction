import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load normal and fault data
normal = pd.read_csv("normal.csv")
fault = pd.read_csv("fault.csv")

# Add labels
normal["label"] = 0  # 0 = Normal
fault["label"] = 1   # 1 = Fault

# Combine datasets
data = pd.concat([normal, fault], ignore_index=True)

# Features and target
X = data[["vib_count", "temp_c", "humidity"]]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"✅ Model trained successfully with accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save model
joblib.dump(model, "motor_model.pkl")
print("💾 Model saved as motor_model.pkl")
