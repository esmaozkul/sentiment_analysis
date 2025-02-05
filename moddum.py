# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:25:31 2025

@author: Esma
"""

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Modeli yükle
model_path = "C:/Users/Esma/Desktop/dersler/sentiment_analysis/sentiment_analysis/emotion_model.keras"
model = load_model(model_path)

# OpenCV'nin yüz tanıma modeli
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 7 farklı duygu etiketi
emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 
                4: "neutral", 5: "sad", 6: "surprise"}

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera açılmadı!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]  # Yüzü kırp
        roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0  # Modelin giriş boyutuna getir ve normalleştir
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

        # Modeli çalıştır
        predictions = model.predict(roi_gray)[0]
        max_index = np.argmax(predictions)  # En yüksek ihtimal hangi duyguda
        emotion = emotion_dict[max_index]

        # Duygu yüzün üstüne yazdır
        cv2.putText(frame, f"Duygun: {emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Terminale duygu bilgisini yazdır
        print(f"Algılanan Duygu: {emotion} ({predictions[max_index]:.2%})")

    cv2.imshow("Duygu Analizi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basınca çık
        break

cap.release()
cv2.destroyAllWindows()
