import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os

# Load trained model
model = tf.keras.models.load_model('models/edge_lstm.keras')

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load real data
df = pd.read_csv('data/plot_00.csv')
features = ['soil_moisture', 'temperature_c', 'rainfall_mm', 'et_mm_day']
for col in features:
    if col not in df.columns:
        df[col] = 0.0

data = df[features].values
data = scaler.transform(data)

SEQ_LEN = 24
sequences = []
for i in range(len(data) - SEQ_LEN):
    sequences.append(data[i:i+SEQ_LEN])
sequences = np.array(sequences[:200], dtype=np.float32)

def representative_data_gen():
    for i in range(len(sequences)):
        yield [sequences[i:i+1]]

# Method 1 — Dynamic range quantization (works with LSTM)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter._experimental_lower_tensor_list_ops = False
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

with open('models/lstm_int8.tflite', 'wb') as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
float32_kb = os.path.getsize('models/edge_lstm.keras') / 1024

print(f"Float32 model size:  {float32_kb:.1f} KB")
print(f"TFLite model size:   {size_kb:.1f} KB")
print(f"Compression:         {(1 - size_kb/float32_kb)*100:.1f}% smaller")
print(f"Saved → models/lstm_int8.tflite")