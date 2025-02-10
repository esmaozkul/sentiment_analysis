# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:22:51 2025

@author: Esma
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    # 📌 Resmi OpenCV ile işle
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 📌 İşlenmiş görüntüyü kaydet
    processed_filename = os.path.join(UPLOAD_FOLDER, "processed_" + file.filename)
    cv2.imwrite(processed_filename, gray)

    return jsonify({"message": "Image processed successfully", "processed_image": processed_filename})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
