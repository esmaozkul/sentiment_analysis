import tensorflow as tf

# Keras modelinizi yükleyin
model = tf.keras.models.load_model('C:/Users/Esma/Desktop/dersler/sentiment_analysis/sentiment_analysis/emotion_model.keras')

# Modeli TensorFlow Lite formatına dönüştürün
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Dönüştürülen modeli .tflite formatında kaydedin
with open('emotion_model.tflite', 'wb') as f:
    f.write(tflite_model)
# -*- coding: utf-8 -*-

