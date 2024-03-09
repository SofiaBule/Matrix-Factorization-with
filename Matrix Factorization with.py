import numpy as np
import tensorflow as tf

# Example dataset (user-item interactions matrix)
interactions = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 1]])

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(4)
])

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(interactions, interactions, epochs=10)

# Example usage:
user_id = 0
item_id = 3
pred = model.predict(np.array([[user_id, item_id]]))
print("Prediction for user", user_id, "and item", item_id, ":", pred)
