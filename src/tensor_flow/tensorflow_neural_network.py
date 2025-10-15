import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("ds_breast_cancer.csv")  # make sure this file is in the same folder

# Detect the label column (M = malignant, B = benign)
label_col = None
for col in data.columns:
    if data[col].astype(str).str.lower().isin(['m', 'b']).any():
        label_col = col
        break

# Encode labels: M = 1, B = 0
data['label'] = data[label_col].apply(lambda x: 1 if str(x).lower() == 'm' else 0)

# Features (numeric columns only)
X = data.select_dtypes(include=['number']).drop(columns=['label'], errors='ignore')
y = data['label']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Neural Network
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),  
    tf.keras.layers.Dense(16, activation='relu'),  
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

# Predictions
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
