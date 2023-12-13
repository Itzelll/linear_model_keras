# adsoft
import numpy as np
import os
import pandas as pd  # Add import for pandas

# TensorFlow
import tensorflow as tf


def circulo(num_datos=500000, R=1, centro_lat=0, centro_lon=0):
    pi = np.pi
    theta = np.random.uniform(0, 2 * pi, size=num_datos)

    # Genera valores positivos para el radio
    r_positive = np.abs(R * np.sqrt(np.random.normal(0, 1, size=num_datos)**2))

    # Calcula las coordenadas x e y
    x = np.cos(theta) * r_positive + centro_lon
    y = np.sin(theta) * r_positive + centro_lat

    # Ajusta la precisión de las coordenadas
    x = np.round(x, 6)
    y = np.round(y, 6)

    # Crea un DataFrame con las coordenadas
    df = pd.DataFrame({'lat': y, 'lon': x})
    return df

#coordenadas zimbabue: -18.962069424880788, 29.921221655885887
datos_zimbabue = circulo(num_datos=100, R=2, centro_lat=-18.962069424880788, centro_lon=29.921221655885887)
#coordenadas liberia: centro_lat=6.446101033163296, centro_lon=-9.618491083388509
datos_liberia = circulo(num_datos=100, R=0.5, centro_lat=6.446101033163296, centro_lon=-9.618491083388509)

# Generate circular data using the circulo function
#circulo_data = circulo(num_datos=800, centro_lat=0, centro_lon=0)
X = np.concatenate([datos_zimbabue, datos_liberia])
X = np.round(X, 6)
y = np.concatenate([np.zeros(800), np.ones(100), np.ones(100)])  # Assign labels (0 for circular data, 1 for Brasilia and Kazakhstan)

train_end = int(0.6 * len(X))
test_start = int(0.8 * len(X))
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=[2], activation='relu', name='cero'),
    tf.keras.layers.Dense(units=4, activation='relu', name='uno'),
    tf.keras.layers.Dense(units=8, activation='relu', name='dos'),
    tf.keras.layers.Dense(units=1, activation='sigmoid', name='tres')
])

linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
print(linear_model.summary())

# Train the model
linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300)

# Predict labels for all data points
all_predictions = linear_model.predict(X).tolist()


# GPS points for zimbabue and liberia
gps_points_zimbabue = [
    [-19.434146197912717, 29.800372205317895],
    [-18.337506598016812, 29.92122191640074],
    [-19.021802145433597, 29.77839965850792],
    [-18.69430959476436, 28.78413701702325],
    [-18.668290815636265, 29.344439721064337]]

gps_points_liberia = [
    [6.1413119177375455, -9.315178367889983],
    [6.185685877093672, -9.26848649566308],
    [6.20616491354859, -9.34127091555077],
    [6.268962317744356, -9.230034349307319],
    [6.165206047278638, -9.353630534022264]]

# Extract predictions for zimbabue and liberia
predictions_zimbabue = linear_model.predict(gps_points_zimbabue).tolist()
predictions_liberia = linear_model.predict(gps_points_liberia).tolist()

print("\nPredictions for zimbabue:")
print(predictions_zimbabue)

print("\nPredictions for liberia:")
print(predictions_liberia)

export_path = 'linear-model/1/'
tf.saved_model.save(linear_model, os.path.join('./',export_path))

