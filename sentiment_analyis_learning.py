import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Veri yolları
train_dir = r"C:/Users/Esma/Desktop/dersler/sentiment_analysis/fer_2013/train"
test_dir = r"C:/Users/Esma/Desktop/dersler/sentiment_analysis/fer_2013/test"

# Duygu sınıfları
emotion_dict = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

# Veriyi yüklemek için yardımcı fonksiyon
def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):  # Her klasör bir duygu sınıfıdır
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Gri tonlama
                img = cv2.resize(img, (48, 48))  # Görüntüyü 48x48'e yeniden boyutlandır
                images.append(img)
                # Label'ı emotion_dict'e göre buluyoruz
                try:
                    labels.append(list(emotion_dict.values()).index(label.lower()))
                except ValueError:
                    print(f"Uyarı: '{label}' etiketi emotion_dict içinde bulunamadı!")
    return np.array(images), np.array(labels)

# Eğitim ve test verilerini yükleme
x_train, y_train = load_data(train_dir)
x_test, y_test = load_data(test_dir)

# Verileri normalize etme (0-1 aralığına çekme)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Verilere uygun şekil kazandırma
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

# Etiketleri one-hot encoding ile dönüştürme
y_train = to_categorical(y_train, num_classes=len(emotion_dict))
y_test = to_categorical(y_test, num_classes=len(emotion_dict))

# Model mimarisi oluşturma
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation="relu"))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(len(emotion_dict), activation="softmax"))

# Modeli derleme
emotion_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0001, decay=1e-6),
    metrics=["accuracy"],
)

# Modeli eğitme
emotion_model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=50,
    validation_split=0.2,  # Eğitim verisinin %20'sini doğrulama için ayırırız.
    shuffle=True,
)

# Modeli JSON dosyasına kaydetme
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# Eğitilmiş model ağırlıklarını .h5 dosyasına kaydetme
emotion_model.save_weights("emotion_model.keras")
print("Model ve ağırlıklar kaydedildi.")

# Modeli test etme
test_loss, test_accuracy = emotion_model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")
