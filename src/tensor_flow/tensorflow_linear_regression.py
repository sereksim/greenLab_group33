import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
data = pd.read_csv("ds_california_housing.csv")
data = data.dropna()

# Fix 'ocean_proximity' column
if 'ocean_proximity' in data.columns:
    data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

# Features and target
X = data.drop(columns=['median_house_value'])
y = data['median_house_value']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build TensorFlow model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mse')

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Make predictions
y_pred = model.predict(X_test).flatten()

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")
